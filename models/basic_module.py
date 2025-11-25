import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import MinkowskiEngine as ME
import pytorch3d.ops
from pytorch3d.ops import knn_points, knn_gather

from data_processing.data_utils import isin, istopk
from models.entropy_model import EntropyBottleneck, SymmetricConditional

def make_layer(block, block_layers, channels):
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))

    return torch.nn.Sequential(*layers)

def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits

def sinusoidal_embedding(values, dim=256, max_period=64):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim // 2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

# Transform
class Encoder_G(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=1,
            out_channels=channels//2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels//2,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=channels)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=channels)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, lmb_input):
        out0 = self.relu(self.down0(self.relu(self.conv0(x))))
        out0 = self.block0([out0, lmb_input])[0]
        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = self.block1([out1, lmb_input])[0]

        return [out1, out0]

class Decoder_G(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=channels)

        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=channels)

        self.conv1_cls = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.feature_fusion = FeatureFusion(channels=channels, inchannels=channels, midchannels=channels)
        self.pruning = ME.MinkowskiPruning()
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training:
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else:
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    def forward(self, x, down, lmb_input, nums_list, ground_truth_list, training=True):
        out = x

        # fusion
        out = self.feature_fusion(out, down)

        out_result_list = []
        #
        out = self.relu(self.conv0(self.relu(self.up0(out))))
        out = self.block0([out, lmb_input])[0]
        out_cls_0 = self.conv0_cls(out)
        out = self.prune_voxel(out, out_cls_0,
                               nums_list[0], ground_truth_list[0], training)
        out_result_list.append(out)
        #
        out = self.relu(self.conv1(self.relu(self.up1(out))))
        out = self.block1([out, lmb_input])[0]
        out_cls_1 = self.conv1_cls(out)
        out = self.prune_voxel(out, out_cls_1,
                               nums_list[1], ground_truth_list[1], training)
        out_result_list.append(out)

        out_cls_list = [out_cls_0, out_cls_1]

        return out_cls_list, out, out_result_list

class Encoder_A(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=channels)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=channels)

        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, lmb_input):
        out = x
        out = self.relu(self.down0(self.relu(self.conv0(out))))
        out = self.block0([out, lmb_input])[0]
        out = self.relu(self.down1(self.relu(self.conv1(out))))
        out = self.block1([out, lmb_input])[0]

        return out

class Decoder_A(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.up0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=channels)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv1 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=64)

        self.deconv3 = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.feature_fusion = FeatureFusion(channels=channels)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

        self.up = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)

    def forward(self, x, x_gpcc, gpcc_color, lmb_input):
        out = x

        x_gpcc = self.sort(x_gpcc, out.C)
        x_gpcc = ME.SparseTensor(features=x_gpcc.F,
                                 coordinate_map_key=out.coordinate_map_key,
                                 coordinate_manager=out.coordinate_manager,
                                 device=out.device)

        out = self.feature_fusion(out, x_gpcc)

        out = self.relu(self.deconv0(self.relu(self.up0(out))))
        out = self.block0([out, lmb_input])[0]
        out = self.relu(self.deconv1(self.relu(self.up1(out))))
        out = self.block1([out, lmb_input])[0]

        out = self.deconv3(out)

        up = self.up(self.up(gpcc_color))

        out = out + up

        return out

class ChannelTransform(torch.nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super().__init__()
        self.channel_transform = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.channel_transform(x)

        return out

class LambdaTransform(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.MAX_LMB = 8192
        self._sin_period = 64

        self.dim = channels
        self.linear0 = torch.nn.Linear(128, 128)
        self.linear1 = torch.nn.Linear(128, 128)
        self.relu = nn.ReLU()

    def forward(self, qs):
        lmb_input = torch.log(torch.tensor([qs]).float()) * self._sin_period / math.log(self.MAX_LMB)
        lmb_input = sinusoidal_embedding(lmb_input, dim=self.dim).cuda()
        lmb_input = self.linear1(self.relu(self.linear0(lmb_input)))

        return lmb_input

class AdaptiveLayer(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.linear = torch.nn.Linear(128, channels * 2)

    def forward(self, x, lmb_input):
        out = x
        elmb = self.linear(lmb_input)

        x_F = out.F
        x_F = x_F * elmb[:, :self.channels] + elmb[:, self.channels:]

        out = ME.SparseTensor(
            features=x_F,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
            device=out.device)
        return out

class AdaptiveInceptionResNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)

        self.adaptive = AdaptiveLayer(channels=channels)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

    def forward(self, input):
        x = input[0]
        lmb_input = input[1]

        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out2 = ME.cat(out0, out1)

        out = self.conv3(self.relu(self.conv2(self.adaptive(out2, lmb_input)))) + x

        return [out, lmb_input]

class InceptionResNet(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x

        return out

class SIA(torch.nn.Module):
    def __init__(self, channels=128, inchannels=1):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=inchannels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=3)
        self.block = make_layer(
            block=AdaptiveInceptionResNet,
            block_layers=3,
            channels=channels)

        self.pct = PointCloudTransformer(channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

    def forward(self, gpcc, lmb_input):
        out = gpcc
        out = self.relu(self.conv(out))
        out = self.block([out, lmb_input])[0]
        out = self.pct(out)

        return out

class target_knn(nn.Module):
    def __init__(self, channels, k=16):
        super(target_knn, self).__init__()
        self.SA0 = self_attention(channels)
        self.SA1 = self_attention(channels)
        self.k = k
        self.Linear = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.pos0 = nn.Sequential(
            nn.Linear(3, channels)
        )
        self.pos1 = nn.Sequential(
            nn.Linear(3, channels)
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.bn0 = ME.MinkowskiBatchNorm(channels)
        self.bn1 = ME.MinkowskiBatchNorm(channels)

    def forward(self, x_reference, x_lossy):
        x_reference_C = x_reference.C.unsqueeze(0).float()
        x_reference_F = x_reference.F.unsqueeze(0).float()

        x_lossy_C = x_lossy.C.unsqueeze(0).float()
        dist, idx, _ = knn_points(x_lossy_C, x_reference_C, K=self.k, return_nn=False, return_sorted=True)
        x_lossy_xyz = x_lossy_C.squeeze(0)[:, 1:]

        x_reference_neibor = knn_gather(x_reference_C[:, :, 1:], idx).squeeze(0)
        x_reference_neibor = x_lossy_xyz[:, None, :] - x_reference_neibor[:, :, :]
        x_reference_neibor_pos_emb = self.pos0(x_reference_neibor)

        # knn
        new_feature = knn_gather(x_reference_F, idx).squeeze(0)
        out = self.SA0(x_lossy, x_reference_neibor_pos_emb, new_feature)  # self-attention
        x_reference_neibor_pos_emb_1 = self.pos1(x_reference_neibor)
        out = self.SA1(out, x_reference_neibor_pos_emb_1, new_feature)  # self-attention

        # skip-connettion
        out = self.bn0(x_lossy + out)
        out_1 = self.Linear(out)
        out = self.bn1(out_1 + out)

        return out

class PointCloudTransformer(nn.Module):
    def __init__(self, input_channel):
        super(PointCloudTransformer, self).__init__()
        self.SA0 = self_attention(input_channel)
        self.SA1 = self_attention(input_channel)
        self.Linear = ME.MinkowskiConvolution(
            in_channels=input_channel,
            out_channels=input_channel,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.pos0 = nn.Sequential(
            nn.Linear(3, input_channel)
        )
        self.pos1 = nn.Sequential(
            nn.Linear(3, input_channel)
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.bn0 = ME.MinkowskiBatchNorm(input_channel)
        self.bn1 = ME.MinkowskiBatchNorm(input_channel)

    def forward(self, x):
        input = x
        out = x
        # knn
        x_C = out.C.unsqueeze(0).float()
        x_F = out.F.unsqueeze(0)
        dist, idx, _ = knn_points(x_C, x_C, K=16, return_nn=False, return_sorted=True)

        # dist, idx, _ = knn_points(x_C, x_C, K=16)
        xyz = x_C.squeeze(0)[:, 1:]
        new_xyz = knn_gather(x_C[:, :, 1:], idx).squeeze(0)
        new_xyz = xyz[:, None, :] - new_xyz[:, :, :]

        xyz_enc = self.pos0(new_xyz)
        new_feature = knn_gather(x_F, idx).squeeze(0)

        # self-attention
        out = self.SA0(out, xyz_enc, new_feature)

        # knn
        out_F = out.F.unsqueeze(0).float()
        new_feature = knn_gather(out_F, idx).squeeze(0)
        xyz_enc_1 = self.pos1(new_xyz)

        # self-attention
        out = self.SA1(out, xyz_enc_1, new_feature)

        # skip-connettion
        out = self.bn0(input + out)
        out_1 = self.Linear(out)
        out = self.bn1(out_1 + out)

        return out

class self_attention(nn.Module):
    def __init__(self, channels):
        super(self_attention, self).__init__()
        self.q_conv = torch.nn.Linear(channels, channels)
        self.k_conv = torch.nn.Linear(channels, channels)
        self.v_conv = torch.nn.Linear(channels, channels)
        self.d = math.sqrt(channels)

    def forward(self, x, xyz_enc, new_feature):
        out = x
        x_q = out.F

        Q = self.q_conv(x_q)
        new_feature = new_feature + xyz_enc
        K = self.k_conv(new_feature)
        K = K.permute(0, 2, 1)
        attention_map = torch.einsum('ndk,nd->nk', K, Q)
        attention_map = F.softmax(attention_map / self.d, dim=-1)
        V = self.v_conv(new_feature)
        attention_feature = torch.einsum('nk,nkd->nd', attention_map, V)
        out = ME.SparseTensor(features=attention_feature, coordinate_map_key=out.coordinate_map_key,
                            coordinate_manager=out.coordinate_manager)
        return out

class MotionFeatureExtraction(torch.nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels*2,
            out_channels=channels*2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels*2,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

    def merge_two_frames(self, f1, f2, stride):
        f1_ = ME.SparseTensor(torch.cat([f1.F, torch.zeros_like(f1.F)], dim=-1), coordinates=f1.C,
                              tensor_stride=stride, device=f1.device)
        f2_ = ME.SparseTensor(torch.cat([torch.zeros_like(f2.F), f2.F], dim=-1), coordinates=f2.C,
                              tensor_stride=stride, coordinate_manager=f1_.coordinate_manager, device=
                              f1.device)

        merged_f = f1_ + f2_

        merged_f = ME.SparseTensor(merged_f.F, coordinates=merged_f.C, tensor_stride=stride,
                                   device=merged_f.device)

        return merged_f

    def forward(self, ref, pred, stride):
        merged_f = self.merge_two_frames(ref, pred, stride)
        out = self.relu(self.conv0(merged_f))
        out = self.conv1(out)
        mask = isin(out.C, pred.C)
        pred_out = self.pruning(out, mask)
        pred_out = self.sort(pred_out, pred.C)
        pred_out = ME.SparseTensor(features=pred_out.F,
                                   coordinate_map_key=pred.coordinate_map_key,
                                   coordinate_manager=pred.coordinate_manager, device=pred.device)

        return pred_out

class InterFusion(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=3 * channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x0, x1, x2):
        out = x0
        out_F = torch.cat((x0.F, x1.F, x2.F), dim=1)
        out = ME.SparseTensor(features=out_F, coordinate_map_key=out.coordinate_map_key,
                              coordinate_manager=out.coordinate_manager)

        out = self.conv0(out)

        return out

class FeatureFusion(torch.nn.Module):
    def __init__(self, channels=128, inchannels=128, midchannels=128, only_fusion=False):
        super().__init__()
        self.only_fusion = only_fusion
        if not self.only_fusion:
            self.conv0 = ME.MinkowskiConvolution(
                in_channels=inchannels,
                out_channels=midchannels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)
            self.conv1 = ME.MinkowskiConvolution(
                in_channels=midchannels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, x_gpcc):
        out = x
        out_g = x_gpcc
        if not self.only_fusion:
            out_g = self.relu(self.conv1(self.conv0(out_g)))

        out_F = torch.cat((out.F, out_g.F), dim=1)
        out = ME.SparseTensor(features=out_F, coordinate_map_key=out.coordinate_map_key,
                              coordinate_manager=out.coordinate_manager)

        out = self.conv2(out)

        return out

# Entropy Model
class HyperEncoder(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv_in = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv_in(x))
        out = self.relu(self.conv0_0(self.conv0(out)))
        out = self.conv1_0(self.conv1(out))

        return out

class HyperDecoder(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.deconv0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv0_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.deconv1 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.deconv1_0 = ME.MinkowskiConvolutionTranspose(
            in_channels=channels * 2,
            out_channels=channels * 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.deconv_out = ME.MinkowskiConvolutionTranspose(
            in_channels=channels * 2,
            out_channels=channels * 2,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.deconv0_0(self.deconv0(x)))
        out = self.relu(self.deconv1_0(self.deconv1(out)))
        out = self.deconv_out(out)

        return out

class InterEntropyModelChannelwiseCheckboardGroupA(torch.nn.Module):
    def __init__(self, channels=128, num_slices=[64, 64]):
        super(InterEntropyModelChannelwiseCheckboardGroupA, self).__init__()

        self.entropy_bottleneck_0 = ContextGroupEntropyModelA(channels, num_slices)
        self.entropy_bottleneck_1 = ContextGroupEntropyModelA(channels, num_slices)
        self.entropy_bottleneck_2 = ContextGroupEntropyModelA(channels, num_slices)
        self.entropy_bottleneck_3 = ContextGroupEntropyModelA(channels, num_slices)

    def forward(self, y, previous_y_tilde, qs, training, eval=False):
        y_tilde_3, z_tilde_3, y_likelihood_3, z_likelihood_3 = self.entropy_bottleneck_3(y, previous_y_tilde,
                                                                                            training)
        y_tilde_2, z_tilde_2, y_likelihood_2, z_likelihood_2 = self.entropy_bottleneck_2(y, previous_y_tilde,
                                                                                            training)
        y_tilde_1, z_tilde_1, y_likelihood_1, z_likelihood_1 = self.entropy_bottleneck_1(y, previous_y_tilde,
                                                                                            training)
        y_tilde_0, z_tilde_0, y_likelihood_0, z_likelihood_0 = self.entropy_bottleneck_0(y, previous_y_tilde,
                                                                                            training)

        bits_3 = get_bits(y_likelihood_3) / float(y.__len__())
        bits_2 = get_bits(y_likelihood_2) / float(y.__len__())
        bits_1 = get_bits(y_likelihood_1) / float(y.__len__())
        bits_0 = get_bits(y_likelihood_0) / float(y.__len__())

        y_tilde_list = [y_tilde_0, y_tilde_1, y_tilde_2, y_tilde_3]
        y_likelihood_list = [y_likelihood_0, y_likelihood_1, y_likelihood_2, y_likelihood_3]

        bits_list = [bits_0.item(), bits_1.item(), bits_2.item(), bits_3.item()]
        index = bits_list.index(min(bits_list))

        y_tilde = y_tilde_list[index]
        y_likelihood = y_likelihood_list[index]

        return y_tilde, 0, y_likelihood, 0, index

class ContextGroupEntropyModelA(torch.nn.Module):
    def __init__(self, channels=128, num_slices=[64, 64]):
        super(ContextGroupEntropyModelA, self).__init__()
        self.num_slices = num_slices
        self.channel = channels
        self.context_model = ContextModel(channels)
        #ChannelContextModel
        self.conditional_entropy_models = torch.nn.ModuleList()
        self.cc_transforms = torch.nn.ModuleList()
        self.sum_slice = 0
        for i, num_slice in enumerate(self.num_slices):
            self.conditional_entropy_models.append(SymmetricConditional())
            if i == 0:
                self.cc_transforms.append(ChannelContextModel(channels=channels, output_channel=num_slice, num_slice=0))
                self.sum_slice += num_slice
            else:
                self.cc_transforms.append(
                    ChannelContextModel(channels=channels, output_channel=num_slice, num_slice=self.sum_slice))
                self.sum_slice += num_slice

    def channel_wise_forward(self, previous_y_tilde, y, prior, training):
        y_slices_F = torch.split(y.F, self.num_slices, dim=-1)
        y_hat_slices = []
        y_likelihoods = []
        y_noise_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            # SparseTensor
            if slice_index == 0:
                support_channel = torch.tensor([], device='cuda')
                prior_slice = self.cc_transforms[slice_index](previous_y_tilde, prior, support_channel)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)
                y_noise_F, y_slice_likelihood = self.conditional_entropy_models[slice_index](y_slice_F, loc, scale,quantize_mode="noise" if training else "symbols")
                y_hat_slice_F = self.conditional_entropy_models[slice_index]._quantize(y_slice_F, mode="symbols")
                y_likelihoods.append(y_slice_likelihood)

                y_hat_slices.append(y_hat_slice_F)
                y_noise_slices.append(y_noise_F)
            else:
                support_channel = torch.cat(y_hat_slices, dim=1)
                prior_slice = self.cc_transforms[slice_index](previous_y_tilde, prior, support_channel)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)
                y_noise_F, y_slice_likelihood = self.conditional_entropy_models[slice_index](y_slice_F, loc, scale,
                                                                                             quantize_mode="noise" if training else "symbols")
                y_hat_slice_F = self.conditional_entropy_models[slice_index]._quantize(y_slice_F, mode="symbols")
                y_likelihoods.append(y_slice_likelihood)
                y_hat_slices.append(y_hat_slice_F)
                y_noise_slices.append(y_noise_F)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihood = torch.cat(y_likelihoods, dim=1)
        y_tilde = ME.SparseTensor(features=y_hat, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        return y_tilde, y_likelihood

    def forward(self, y, previous_y_tilde, training):
        prior = self.context_model(previous_y_tilde)
        y_tilde, y_likelihood = self.channel_wise_forward(previous_y_tilde, y, prior, training)
        _ = torch.ones_like(y.F).cuda()

        return y_tilde, _, y_likelihood, _

class ChannelContextModel(torch.nn.Module):
    def __init__(self, channels=128, output_channel=64, num_slice=0):
        super(ChannelContextModel, self).__init__()
        self.channels = channels
        self.conv0 = ME.MinkowskiConvolution(in_channels=channels * 4 + num_slice,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=output_channel * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, previous, current, support_channel):
        context = previous
        if len(support_channel) == 0:
            context = context
        else:
            context = ME.SparseTensor(
                features=torch.cat((context.F, support_channel), dim=-1),
                coordinate_map_key=context.coordinate_map_key,
                coordinate_manager=context.coordinate_manager,
                device=context.device)
        out = self.relu(self.conv0(context))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)

        return out

class InterEntropyModelChannelwiseCheckboardGroup(torch.nn.Module):
    def __init__(self, channels=128, num_slices=[64, 64]):
        super(InterEntropyModelChannelwiseCheckboardGroup, self).__init__()
        self.channel = channels
        self.entropy_bottleneck_0 = ContextGroupEntropyModel(channels, num_slices)
        self.entropy_bottleneck_1 = ContextGroupEntropyModel(channels, num_slices)
        self.entropy_bottleneck_2 = ContextGroupEntropyModel(channels, num_slices)
        self.entropy_bottleneck_3 = ContextGroupEntropyModel(channels, num_slices)

    def forward(self, y, previous_group, previous_y_tilde, qs, training, eval=False):
        y_tilde_3, z_tilde_3, y_likelihood_3, z_likelihood_3 = self.entropy_bottleneck_3(y, previous_group, previous_y_tilde,
                                                                                            training)
        y_tilde_2, z_tilde_2, y_likelihood_2, z_likelihood_2 = self.entropy_bottleneck_2(y, previous_group, previous_y_tilde,
                                                                                            training)
        y_tilde_1, z_tilde_1, y_likelihood_1, z_likelihood_1 = self.entropy_bottleneck_1(y, previous_group, previous_y_tilde,
                                                                                            training)
        y_tilde_0, z_tilde_0, y_likelihood_0, z_likelihood_0 = self.entropy_bottleneck_0(y, previous_group, previous_y_tilde,
                                                                                            training)

        bits_3 = get_bits(y_likelihood_3) / float(y.__len__())
        bits_2 = get_bits(y_likelihood_2) / float(y.__len__())
        bits_1 = get_bits(y_likelihood_1) / float(y.__len__())
        bits_0 = get_bits(y_likelihood_0) / float(y.__len__())

        y_tilde_list = [y_tilde_0, y_tilde_1, y_tilde_2, y_tilde_3]
        y_likelihood_list = [y_likelihood_0, y_likelihood_1, y_likelihood_2, y_likelihood_3]

        bits_list = [bits_0.item(), bits_1.item(), bits_2.item(), bits_3.item()]
        index = bits_list.index(min(bits_list))

        y_tilde = y_tilde_list[index]
        y_likelihood = y_likelihood_list[index]

        return y_tilde, 0, y_likelihood, 0, index

class ContextGroupEntropyModel(torch.nn.Module):
    def __init__(self, channels=128, num_slices=[16, 16]):
        super(ContextGroupEntropyModel, self).__init__()
        self.num_slices = num_slices
        self.channel = channels
        self.context_model = ContextModel(channels)

        self.conditional_entropy_models = torch.nn.ModuleList()
        self.context_models = torch.nn.ModuleList()
        self.sum_slice = 0
        for i, num_slice in enumerate(self.num_slices):
            self.conditional_entropy_models.append(SymmetricConditional())
            self.context_models.append(ContextModelHyperGroupChannelWise(channels=channels, support_channel=self.sum_slice,
                                                                         output_channel=num_slice *2))
            self.sum_slice += num_slice

    def channel_wise_forward(self, y, last_y, previous_y, prior, training):
        y_slices_F = torch.split(y.F, self.num_slices, dim=-1)
        y_hat_slices = []
        y_likelihoods = []
        y_noise_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            # SparseTensor
            y_slice = ME.SparseTensor(features=y_slice_F, coordinate_map_key=prior.coordinate_map_key,
                                      coordinate_manager=prior.coordinate_manager, device=prior.device)
            if slice_index == 0:
                loc, scale = self.context_models[slice_index](last_y, y_slice, prior, previous_y, y_slice_support=torch.tensor([],device='cuda'))
                scale = torch.clamp(scale, min=1e-8)
                y_noise_F, y_slice_likelihood = self.conditional_entropy_models[slice_index](y_slice_F, loc, scale,
                                                                                             quantize_mode="noise" if training else "symbols")
                y_hat_slice_F = self.conditional_entropy_models[slice_index]._quantize(y_slice_F, mode="symbols")
                y_likelihoods.append(y_slice_likelihood)
                y_hat_slices.append(y_hat_slice_F)
                y_noise_slices.append(y_noise_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                loc, scale = self.context_models[slice_index](last_y, y_slice, prior, previous_y, y_slice_support=y_slice_support)

                scale = torch.clamp(scale, min=1e-8)

                y_noise_F, y_slice_likelihood = self.conditional_entropy_models[slice_index](y_slice_F, loc, scale,
                                                                                             quantize_mode="noise" if training else "symbols")
                y_hat_slice_F = self.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                       mode="symbols")
                y_likelihoods.append(y_slice_likelihood)
                y_hat_slices.append(y_hat_slice_F)
                y_noise_slices.append(y_noise_F)

        y_hat = torch.cat(y_noise_slices, dim=1)
        y_likelihood = torch.cat(y_likelihoods, dim=1)
        y_tilde = ME.SparseTensor(features=y_hat, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        return y_tilde, y_likelihood

    def forward(self, y, previous_group, previous_y_tilde, training):
        prior = self.context_model(previous_y_tilde)
        y_tilde, y_likelihood = self.channel_wise_forward(y, previous_group, previous_y_tilde, prior, training)
        _ = torch.ones_like(y.F).cuda()

        return y_tilde, _, y_likelihood, _

class ContextModelHyperGroupChannelWise(torch.nn.Module):
    def __init__(self, channels=128, support_channel=8, output_channel=16):
        super(ContextModelHyperGroupChannelWise, self).__init__()
        self.channels = channels
        self.output_channel = output_channel
        self.target_conv = ME.MinkowskiConvolution(in_channels=channels,
                                        out_channels=channels,
                                        kernel_size=5,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)
        self.conv = ME.MinkowskiConvolution(in_channels=channels * 4 + support_channel,
                                        out_channels=channels * 4,
                                        kernel_size=3,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)
        self.conv_in = ME.MinkowskiConvolution(in_channels=channels * 4,
                                        out_channels=channels * 4,
                                        kernel_size=3,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)

        self.conv0 = ME.MinkowskiConvolution(in_channels=channels*4,
                                            out_channels=channels*3,
                                            kernel_size=3,
                                            stride=1,
                                            bias=True,
                                            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels*3)
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels*3,
                                            out_channels=channels*2,
                                            kernel_size=3,
                                            stride=1,
                                            bias=True,
                                            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels*2)

        self.conv2 = ME.MinkowskiConvolution(in_channels=channels*2,
                                            out_channels=output_channel,
                                            kernel_size=3,
                                            stride=1,
                                            bias=True,
                                            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def knn_interpolation(self, x, new_coords, k):
        new_coords_C = new_coords.C[:, 1:].unsqueeze(0).float()
        x_coords_C = x.C[:, 1:].unsqueeze(0).float()
        x_attr = x.F.unsqueeze(0).float()
        x_nn = pytorch3d.ops.knn_points(new_coords_C, x_coords_C, K=k)
        knn_attribute = pytorch3d.ops.knn_gather(x_attr[:, :, :], x_nn.idx).squeeze(0)

        interpolation_attribute = torch.mean(knn_attribute, dim=1)
        return interpolation_attribute

    def forward(self, x_gp1, x_gp2,  hyper, previous_y, y_slice_support):
        context = self.target_conv(x_gp1, hyper.C)
        context = ME.SparseTensor(
            features=context.F,
            coordinate_map_key=hyper.coordinate_map_key,
            coordinate_manager=hyper.coordinate_manager,
            device=hyper.device)
        interpolation_feature = self.knn_interpolation(x_gp1, x_gp2, k=1)
        interpolation_feature = ME.SparseTensor(
            features=interpolation_feature, coordinates=hyper.C,
            device=hyper.device)

        if context.coordinate_manager == interpolation_feature.coordinate_manager:
            context_hyper = ME.cat(context, hyper)
            context_hyper = ME.cat(context_hyper, interpolation_feature)
        else:
            context_hyper = ME.SparseTensor(
                features=torch.cat((context.F, hyper.F), dim=-1),
                coordinate_map_key=hyper.coordinate_map_key,
                coordinate_manager=hyper.coordinate_manager,
                device=hyper.device)

            context_hyper = ME.SparseTensor(
                features=torch.cat((context_hyper.F, interpolation_feature.F), dim=-1),
                coordinate_map_key=context_hyper.coordinate_map_key,
                coordinate_manager=context_hyper.coordinate_manager,
                device=context_hyper.device)
        if len(y_slice_support) is 0:
            context_hyper = context_hyper
        else:
            context_hyper = ME.SparseTensor(
                features=torch.cat((context_hyper.F, y_slice_support), dim=-1),
                coordinate_map_key=context_hyper.coordinate_map_key,
                coordinate_manager=context_hyper.coordinate_manager,
                device=context_hyper.device)
        context = self.conv_in(self.relu(self.conv(context_hyper)))

        out = self.relu(self.block0(self.conv0(context)))
        out = self.relu(self.block1(self.conv1(out)))
        out = self.conv2(out)
        params = out.F
        loc = params[:, :self.output_channel // 2]
        scale= params[:, self.output_channel // 2:]

        return loc, scale.abs()

class ContextModel(torch.nn.Module):
    def __init__(self, channels=128):
        super(ContextModel, self).__init__()
        self.channels = channels
        self.conv0 = ME.MinkowskiConvolution(in_channels=channels * 4,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.conv2 = ME.MinkowskiConvolution(in_channels=channels * 2,
                                             out_channels=channels * 2,
                                             kernel_size=1,
                                             stride=1,
                                             bias=True,
                                             dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, prior):
        out = self.relu(self.conv0(prior))
        out = self.relu(self.conv1(out))
        out = self.conv2(out)

        return out

class HyperChannelwiseEntropyModelGroup(torch.nn.Module):
    def __init__(self, channels=128, num_slices=[64,64]):
        super(HyperChannelwiseEntropyModelGroup, self).__init__()
        self.entropy_bottleneck_0 = HyperChannelwiseEntropyModel(channels, num_slices)
        self.entropy_bottleneck_1 = HyperChannelwiseEntropyModel(channels, num_slices)
        self.entropy_bottleneck_2 = HyperChannelwiseEntropyModel(channels, num_slices)
        self.entropy_bottleneck_3 = HyperChannelwiseEntropyModel(channels, num_slices)

    def forward(self, y, qs, training, eval=False):
        y_tilde_3, z_tilde_3, y_likelihood_3, z_likelihood_3 = self.entropy_bottleneck_3(y, training)
        y_tilde_2, z_tilde_2, y_likelihood_2, z_likelihood_2 = self.entropy_bottleneck_2(y, training)
        y_tilde_1, z_tilde_1, y_likelihood_1, z_likelihood_1 = self.entropy_bottleneck_1(y, training)
        y_tilde_0, z_tilde_0, y_likelihood_0, z_likelihood_0 = self.entropy_bottleneck_0(y, training)

        bits_3 = (get_bits(y_likelihood_3) + get_bits(z_likelihood_3)) / float(y.__len__())
        bits_2 = (get_bits(y_likelihood_2) + get_bits(z_likelihood_2)) / float(y.__len__())
        bits_1 = (get_bits(y_likelihood_1) + get_bits(z_likelihood_1)) / float(y.__len__())
        bits_0 = (get_bits(y_likelihood_0) + get_bits(z_likelihood_0)) / float(y.__len__())

        y_tilde_list = [y_tilde_0, y_tilde_1, y_tilde_2, y_tilde_3]
        z_tilde_list = [z_tilde_0, z_tilde_1, z_tilde_2, z_tilde_3]
        y_likelihood_list = [y_likelihood_0, y_likelihood_1, y_likelihood_2, y_likelihood_3]
        z_likelihood_list = [z_likelihood_0, z_likelihood_1, z_likelihood_2, z_likelihood_3]

        bits_list = [bits_0.item(), bits_1.item(), bits_2.item(), bits_3.item()]
        index = bits_list.index(min(bits_list))

        y_tilde = y_tilde_list[index]
        z_tilde = z_tilde_list[index]
        y_likelihood = y_likelihood_list[index]
        z_likelihood = z_likelihood_list[index]


        return y_tilde, z_tilde, y_likelihood, z_likelihood, index

class HyperChannelwiseEntropyModel(torch.nn.Module):
    def __init__(self, channels=128, num_slices=[64, 64]):
        super(HyperChannelwiseEntropyModel, self).__init__()
        self.channel = channels
        self.num_slices = num_slices
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.hyper_encoder = HyperEncoder(channels)
        self.hyper_decoder = HyperDecoder(channels)
        self.conditional_entropy_model = SymmetricConditional()
        self.conditional_entropy_models = torch.nn.ModuleList()
        self.cc_transforms = torch.nn.ModuleList()
        self.sum_slice = 0
        for i, num_slice in enumerate(self.num_slices):
            self.conditional_entropy_models.append(SymmetricConditional())
            if i == 0:
                self.cc_transforms.append(SliceTransform(channels=channels, output_channel=num_slice, num_slice=0))
                self.sum_slice += num_slice
            else:
                self.cc_transforms.append(
                    SliceTransform(channels=channels, output_channel=num_slice, num_slice=self.sum_slice))
                self.sum_slice += num_slice
    def channel_wise_forward(self, y,prior, training):
        y_slices_F = torch.split(y.F, self.num_slices, dim=-1)
        y_hat_slices = []
        y_likelihoods = []
        y_noise_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            # SparseTensor
            if slice_index == 0:
                prior_slice = self.cc_transforms[slice_index](prior)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)
                y_noise_F, y_slice_likelihood = self.conditional_entropy_models[slice_index](y_slice_F, loc, scale,quantize_mode="noise" if training else "symbols")
                y_hat_slice_F = self.conditional_entropy_models[slice_index]._quantize(y_slice_F, mode="symbols")
                y_likelihoods.append(y_slice_likelihood)

                y_hat_slices.append(y_hat_slice_F)
                y_noise_slices.append(y_noise_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                prior_support = ME.SparseTensor(features=torch.cat([prior.F, y_slice_support], dim=1), coordinate_map_key=prior.coordinate_map_key,
                                              coordinate_manager=prior.coordinate_manager, device=prior.device)

                prior_slice = self.cc_transforms[slice_index](prior_support)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)
                y_noise_F, y_slice_likelihood = self.conditional_entropy_models[slice_index](y_slice_F, loc, scale,
                                                                                             quantize_mode="noise" if training else "symbols")
                y_hat_slice_F = self.conditional_entropy_models[slice_index]._quantize(y_slice_F, mode="symbols")
                y_likelihoods.append(y_slice_likelihood)
                y_hat_slices.append(y_hat_slice_F)
                y_noise_slices.append(y_noise_F)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihood = torch.cat(y_likelihoods, dim=1)
        y_tilde = ME.SparseTensor(features=y_hat, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        return y_tilde, y_likelihood

    def forward(self, y, training):
        z = self.hyper_encoder(y)
        z_F, z_likelihood = self.entropy_bottleneck(z.F, quantize_mode="noise" if training else "symbols")
        z_tilde = ME.SparseTensor(features=z_F, coordinate_map_key=z.coordinate_map_key,
                                  coordinate_manager=z.coordinate_manager, device=z.device)

        prior = self.hyper_decoder(z_tilde)



        y_tilde, y_likelihood = self.channel_wise_forward(y, prior, training)

        return y_tilde, z_tilde, y_likelihood, z_likelihood

class SliceTransform(torch.nn.Module):
    def __init__(self, channels, output_channel, num_slice, kernel_size=3):
        super(SliceTransform, self).__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels * 2 + num_slice ,
            out_channels=channels * 2 + num_slice,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.IRN0 = InceptionResNet(channels=channels * 2+ num_slice)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels * 2 + num_slice,
            out_channels=channels * 2 + num_slice,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.IRN1 = InceptionResNet(channels=channels * 2 + num_slice)

        self.conv_out = ME.MinkowskiConvolution(
            in_channels=channels * 2 + num_slice,
            out_channels=output_channel * 2,
            kernel_size=kernel_size,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.IRN0(self.relu(self.conv0(x))))
        out = self.relu(self.IRN1(self.relu(self.conv1(out))))
        out = self.conv_out(out)

        return out

class HyperCheckboardChannelwiseEntropyModelGroup(torch.nn.Module):
    def __init__(self, channels=128, num_slices=[64, 64]):
        super(HyperCheckboardChannelwiseEntropyModelGroup, self).__init__()
        self.entropy_bottleneck_0 = HyperCheckboardChannelwiseEntropyModel(channels, num_slices)
        self.entropy_bottleneck_1 = HyperCheckboardChannelwiseEntropyModel(channels, num_slices)
        self.entropy_bottleneck_2 = HyperCheckboardChannelwiseEntropyModel(channels, num_slices)
        self.entropy_bottleneck_3 = HyperCheckboardChannelwiseEntropyModel(channels, num_slices)

    def forward(self, y, last_y, qs, training, eval=False):
        y_tilde_3, z_tilde_3, y_likelihood_3, z_likelihood_3 = self.entropy_bottleneck_3(y, last_y, training)
        y_tilde_2, z_tilde_2, y_likelihood_2, z_likelihood_2 = self.entropy_bottleneck_2(y, last_y, training)
        y_tilde_1, z_tilde_1, y_likelihood_1, z_likelihood_1 = self.entropy_bottleneck_1(y, last_y, training)
        y_tilde_0, z_tilde_0, y_likelihood_0, z_likelihood_0 = self.entropy_bottleneck_0(y, last_y, training)

        bits_3 = (get_bits(y_likelihood_3) + get_bits(z_likelihood_3)) / float(y.__len__())
        bits_2 = (get_bits(y_likelihood_2) + get_bits(z_likelihood_2)) / float(y.__len__())
        bits_1 = (get_bits(y_likelihood_1) + get_bits(z_likelihood_1)) / float(y.__len__())
        bits_0 = (get_bits(y_likelihood_0) + get_bits(z_likelihood_0)) / float(y.__len__())

        y_tilde_list = [y_tilde_0, y_tilde_1, y_tilde_2, y_tilde_3]
        z_tilde_list = [z_tilde_0, z_tilde_1, z_tilde_2, z_tilde_3]
        y_likelihood_list = [y_likelihood_0, y_likelihood_1, y_likelihood_2, y_likelihood_3]
        z_likelihood_list = [z_likelihood_0, z_likelihood_1, z_likelihood_2, z_likelihood_3]

        bits_list = [bits_0.item(), bits_1.item(), bits_2.item(), bits_3.item()]
        index = bits_list.index(min(bits_list))

        y_tilde = y_tilde_list[index]
        z_tilde = z_tilde_list[index]
        y_likelihood = y_likelihood_list[index]
        z_likelihood = z_likelihood_list[index]

        return y_tilde, z_tilde, y_likelihood, z_likelihood, index

class HyperCheckboardChannelwiseEntropyModel(torch.nn.Module):
    def __init__(self, channels=128, num_slices=[64, 64]):
        super(HyperCheckboardChannelwiseEntropyModel, self).__init__()
        self.channel = channels
        self.num_slices = num_slices
        self.entropy_bottleneck = EntropyBottleneck(channels)
        self.hyper_encoder = HyperEncoder(channels)
        self.hyper_decoder = HyperDecoder(channels)
        self.conditional_entropy_models = torch.nn.ModuleList()
        self.context_models = torch.nn.ModuleList()
        self.sum_slice = 0
        for i, num_slice in enumerate(self.num_slices):
            self.conditional_entropy_models.append(SymmetricConditional())
            self.context_models.append(CheckboardContextModel(channels=channels, support_channel=self.sum_slice,
                                                                         output_channel=num_slice *2))
            self.sum_slice += num_slice

    def channel_wise_forward(self, y, last_y, prior, training):
        y_slices_F = torch.split(y.F, self.num_slices, dim=-1)
        y_hat_slices = []
        y_likelihoods = []
        y_noise_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            # SparseTensor
            y_slice = ME.SparseTensor(features=y_slice_F, coordinate_map_key=prior.coordinate_map_key,
                                      coordinate_manager=prior.coordinate_manager, device=prior.device)
            if slice_index == 0:
                loc, scale = self.context_models[slice_index](last_y, y_slice, prior, y_slice_support=torch.tensor([],device='cuda'))
                scale = torch.clamp(scale, min=1e-8)
                y_noise_F, y_slice_likelihood = self.conditional_entropy_models[slice_index](y_slice_F, loc, scale,
                                                                                             quantize_mode="noise" if training else "symbols")
                y_hat_slice_F = self.conditional_entropy_models[slice_index]._quantize(y_slice_F, mode="symbols")
                y_likelihoods.append(y_slice_likelihood)

                y_hat_slices.append(y_hat_slice_F)
                y_noise_slices.append(y_noise_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                loc, scale = self.context_models[slice_index](last_y, y_slice, prior, y_slice_support=y_slice_support)

                scale = torch.clamp(scale, min=1e-8)

                y_noise_F, y_slice_likelihood = self.conditional_entropy_models[slice_index](y_slice_F, loc, scale,
                                                                                             quantize_mode="noise" if training else "symbols")
                y_hat_slice_F = self.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                       mode="symbols")
                y_likelihoods.append(y_slice_likelihood)
                y_hat_slices.append(y_hat_slice_F)
                y_noise_slices.append(y_noise_F)

        y_hat = torch.cat(y_noise_slices, dim=1)
        y_likelihood = torch.cat(y_likelihoods, dim=1)
        y_tilde = ME.SparseTensor(features=y_hat, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)
        return y_tilde, y_likelihood

    def forward(self, y, last_y, training):
        z = self.hyper_encoder(y)
        z_F, z_likelihood = self.entropy_bottleneck(z.F, quantize_mode="noise" if training else "symbols")
        z_tilde = ME.SparseTensor(features=z_F, coordinate_map_key=z.coordinate_map_key,
                                  coordinate_manager=z.coordinate_manager, device=z.device)
        prior = self.hyper_decoder(z_tilde)
        y_tilde, y_likelihood = self.channel_wise_forward(y, last_y, prior, training)


        return y_tilde, z_tilde, y_likelihood, z_likelihood

class CheckboardContextModel(torch.nn.Module):
    def __init__(self, channels=128, support_channel=8, output_channel=16):
        super(CheckboardContextModel, self).__init__()
        self.channels = channels
        self.output_channel = output_channel
        self.target_conv = ME.MinkowskiConvolution(in_channels=channels,
                                        out_channels=channels,
                                        kernel_size=5,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)
        self.conv = ME.MinkowskiConvolution(in_channels=channels * 4 + support_channel,
                                        out_channels=channels * 4,
                                        kernel_size=3,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)
        self.conv_in = ME.MinkowskiConvolution(in_channels=channels * 4,
                                        out_channels=channels * 4,
                                        kernel_size=3,
                                        stride=1,
                                        dilation=1,
                                        bias=True,
                                        dimension=3)

        self.conv0 = ME.MinkowskiConvolution(in_channels=channels*4,
                                            out_channels=channels*3,
                                            kernel_size=3,
                                            stride=1,
                                            bias=True,
                                            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels*3)
        self.conv1 = ME.MinkowskiConvolution(in_channels=channels*3,
                                            out_channels=channels*2,
                                            kernel_size=3,
                                            stride=1,
                                            bias=True,
                                            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=channels*2)

        self.conv2 = ME.MinkowskiConvolution(in_channels=channels*2,
                                            out_channels=output_channel,
                                            kernel_size=3,
                                            stride=1,
                                            bias=True,
                                            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def knn_interpolation(self, x, new_coords, k):
        new_coords_C = new_coords.C[:, 1:].unsqueeze(0).float()
        x_coords_C = x.C[:, 1:].unsqueeze(0).float()
        x_attr = x.F.unsqueeze(0).float()
        x_nn = pytorch3d.ops.knn_points(new_coords_C, x_coords_C, K=k)
        knn_attribute = pytorch3d.ops.knn_gather(x_attr[:, :, :], x_nn.idx).squeeze(0)

        interpolation_attribute = torch.mean(knn_attribute, dim=1)
        return interpolation_attribute

    def forward(self, x_gp1, x_gp2,  hyper, y_slice_support):
        context = self.target_conv(x_gp1, hyper.C)
        context = ME.SparseTensor(
            features=context.F,
            coordinate_map_key=hyper.coordinate_map_key,
            coordinate_manager=hyper.coordinate_manager,
            device=hyper.device)
        interpolation_feature = self.knn_interpolation(x_gp1, x_gp2, k=1)
        interpolation_feature = ME.SparseTensor(
            features=interpolation_feature, coordinates=hyper.C,
            device=hyper.device)

        if context.coordinate_manager == interpolation_feature.coordinate_manager:
            context_hyper = ME.cat(context, hyper)
            context_hyper = ME.cat(context_hyper, interpolation_feature)
        else:
            context_hyper = ME.SparseTensor(
                features=torch.cat((context.F, hyper.F), dim=-1),
                coordinate_map_key=hyper.coordinate_map_key,
                coordinate_manager=hyper.coordinate_manager,
                device=hyper.device)

            context_hyper = ME.SparseTensor(
                features=torch.cat((context_hyper.F, interpolation_feature.F), dim=-1),
                coordinate_map_key=context_hyper.coordinate_map_key,
                coordinate_manager=context_hyper.coordinate_manager,
                device=context_hyper.device)
        if len(y_slice_support) is 0:
            context_hyper = context_hyper
        else:
            context_hyper = ME.SparseTensor(
                features=torch.cat((context_hyper.F, y_slice_support), dim=-1),
                coordinate_map_key=context_hyper.coordinate_map_key,
                coordinate_manager=context_hyper.coordinate_manager,
                device=context_hyper.device)
        context = self.conv_in(self.relu(self.conv(context_hyper)))

        out = self.relu(self.block0(self.conv0(context)))
        out = self.relu(self.block1(self.conv1(out)))
        out = self.conv2(out)
        params = out.F
        loc = params[:, :self.output_channel // 2]
        scale= params[:, self.output_channel // 2:]

        return loc, scale.abs()