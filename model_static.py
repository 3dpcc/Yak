import os
import numpy as np
import torch
import MinkowskiEngine as ME
from models.basic_module import Encoder_G, Decoder_G, Encoder_A, Decoder_A, LambdaTransform, ChannelTransform, HyperCheckboardChannelwiseEntropyModelGroup, HyperChannelwiseEntropyModelGroup, SIA
from data_processing.data_utils import sort_sparse_tensor, load_sparse_tensor, knn_interpolation, rgb2yuv

class PCGCModel(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        # transform
        self.lambda_transform_geo = LambdaTransform(channels)
        self.extractor = SIA(channels=32)
        self.encoder_G = Encoder_G(channels=32)
        self.decoder_G = Decoder_G(channels=32)
        self.channel_transform_enc = ChannelTransform(32, 8)
        self.channel_transform_dec = ChannelTransform(8, 32)

        # entropy model
        self.entropy_bottleneck_1 = HyperChannelwiseEntropyModelGroup(8, [4, 4])
        self.entropy_bottleneck_2 = HyperCheckboardChannelwiseEntropyModelGroup(8, [4, 4])

        # tools
        self.downsampler = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

    def sort_sparse_tensor(self, sparse_tensor):
        """ Sort points in sparse tensor according to their coordinates.
        """
        coords_new = sparse_tensor.C[:,1:] // 4

        mask = (coords_new[:,2] % 2 == 0) & (coords_new[:,1] % 2 == 0) & (coords_new[:,0] % 2 == 0)
        mask2 = (coords_new[:,2] % 2 == 0) & (coords_new[:,1] % 2 == 1) & (coords_new[:,0] % 2 == 1)
        mask3 = (coords_new[:,2] % 2 == 1) & (coords_new[:,1] % 2 == 1) & (coords_new[:,0] % 2 == 0)
        mask4 = (coords_new[:,2] % 2 == 1) & (coords_new[:,1] % 2 == 0) & (coords_new[:,0] % 2 == 1)

        mask_odd = mask | mask2 | mask3 | mask4

        return mask_odd

    def build_sparsetensor(self, coords, feats):
        sparsetensor = ME.SparseTensor(features=feats, coordinates=coords, device='cuda')

        return sparsetensor
    def build_group(self, y):
        indices_sort = self.sort_sparse_tensor(y)
        y_coords_group_list = []
        y_feats_group_list = []
        y_coords_group2_list = []
        y_feats_group2_list = []

        y_coords_group_list.append(y.C[indices_sort == True] // 4)
        y_feats_group_list.append(y.F[indices_sort == True])
        y_coords_group2_list.append(y.C[indices_sort == False] // 4)
        y_feats_group2_list.append(y.F[indices_sort == False])

        y_coords_group = torch.cat(y_coords_group_list, dim=0)
        y_feats_group = torch.cat(y_feats_group_list, dim=0)
        y_coords_group2 = torch.cat(y_coords_group2_list, dim=0)
        y_feats_group2 = torch.cat(y_feats_group2_list, dim=0)

        group = self.build_sparsetensor(y_coords_group, y_feats_group)
        group2 = self.build_sparsetensor(y_coords_group2, y_feats_group2)

        return group, group2

    def quantize(self, inputs):
        from models.entropy_model import RoundNoGradient

        return RoundNoGradient.apply(inputs)

    def inv_group(self, y, group1_tilde, group2_tilde, y_likelihood_1, y_likelihood_2, z_likelihood_1, z_likelihood_2):
        y_likelihoods = []
        z_likelihoods = []

        y_tilde12_C = torch.cat([group1_tilde.C, group2_tilde.C], dim=0)
        y_tilde12_F = torch.cat([group1_tilde.F, group2_tilde.F], dim=0)

        y_tilde12 = ME.SparseTensor(features=y_tilde12_F, coordinates=y_tilde12_C, device='cuda')

        y_likelihoods.append(y_likelihood_1)
        y_likelihoods.append(y_likelihood_2)
        z_likelihoods.append(z_likelihood_1)
        z_likelihoods.append(z_likelihood_2)
        z_likelihood = torch.cat(z_likelihoods, dim=0)
        y_likelihood = torch.cat(y_likelihoods, dim=0)

        y_gp_C = torch.cat([y_tilde12.C * 4], dim=0)
        y_gp_F = torch.cat([y_tilde12.F], dim=0)
        y_gp_new = self.build_sparsetensor(y_gp_C, y_gp_F)

        y_hat = self.sort(y_gp_new, y.C)
        y_tilde = ME.SparseTensor(features=y_hat.F, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)

        return y_tilde, y_likelihood, z_likelihood

    def forward_geo(self, x, qs, training, eval=False):
        x = ME.SparseTensor(
            features=(torch.ones(len(x)).unsqueeze(1)),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)

        lmb_input = self.lambda_transform_geo(qs)

        # Encoder
        y_list = self.encoder_G(x, lmb_input)
        y = y_list[0]

        down = ME.SparseTensor(
            features=(torch.ones(len(y)).unsqueeze(1)),
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)
        down = self.extractor(down, lmb_input)

        ground_truth_list = y_list[1:] + [x]

        nums_list = [[len(C) for C in [ground_truth.C]] \
                     for ground_truth in ground_truth_list]

        y = self.channel_transform_enc(y)

        # Quantizer & Entropy Model
        groupA, groupB = self.build_group(y)
        y_tilde1, _, y_likelihood_1, z_likelihood_1, _ = self.entropy_bottleneck_1(groupA, qs, training, eval)
        y_tilde2, _, y_likelihood_2, z_likelihood_2, _ = self.entropy_bottleneck_2(groupB, y_tilde1, qs, training, eval)

        y_tilde, y_likelihood, z_likelihood = self.inv_group(y, y_tilde1,
                                                        y_tilde2, y_likelihood_1, y_likelihood_2, z_likelihood_1,
                                                        z_likelihood_2)

        y_tilde = self.channel_transform_dec(y_tilde)

        # Decoder
        out_cls_list, out, out_result_list = self.decoder_G(y_tilde, down, lmb_input, nums_list, ground_truth_list,
                                                            training=False)

        return {'out': out,
                'out_cls_list': out_cls_list,
                'y_tilde': y_tilde,
                'likelihood': y_likelihood,
                'z_likelihood': z_likelihood,
                'ground_truth_list': ground_truth_list,
                'out_result_list': out_result_list}

    @torch.no_grad()
    def encode(self, x, qs, filename):
        lmb_input = self.lambda_transform_geo(qs)

        ################################################ Encoder ################################################
        x = sort_sparse_tensor(x)
        x = ME.SparseTensor(
            features=(torch.ones(len(x)).unsqueeze(1)),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)

        # Encoder
        y_list = self.encoder_G(x, lmb_input)
        y = y_list[0]

        ground_truth_list = y_list[1:] + [x]

        nums_list = [[len(C) for C in [ground_truth.C]] \
                     for ground_truth in ground_truth_list]

        y = self.channel_transform_enc(y)

        ################################################ Entropy Model ################################################
        group, group2 = self.build_group(y)
        group_tilde1, _, _, _, index_gp1 = self.entropy_bottleneck_1(group, qs, False, True)
        _, _, _, _, index_gp2 = self.entropy_bottleneck_2(group2, group_tilde1, qs, False, True)

        entropy_model_list1 = [self.entropy_bottleneck_1.entropy_bottleneck_0,
                               self.entropy_bottleneck_1.entropy_bottleneck_1,
                               self.entropy_bottleneck_1.entropy_bottleneck_2,
                               self.entropy_bottleneck_1.entropy_bottleneck_3]
        entropy_model_list2 = [self.entropy_bottleneck_2.entropy_bottleneck_0,
                               self.entropy_bottleneck_2.entropy_bottleneck_1,
                               self.entropy_bottleneck_2.entropy_bottleneck_2,
                               self.entropy_bottleneck_2.entropy_bottleneck_3]
        entropy_model_group1 = entropy_model_list1[index_gp1]
        entropy_model_group2 = entropy_model_list2[index_gp2]

        # group1 hyper
        z_gp1 = entropy_model_group1.hyper_encoder(group)

        z_gp1_F = entropy_model_group1.entropy_bottleneck._quantize(z_gp1.F, mode="symbols")
        z_gp1_tilde = ME.SparseTensor(features=z_gp1_F, coordinate_map_key=z_gp1.coordinate_map_key,
                                      coordinate_manager=z_gp1.coordinate_manager, device=z_gp1.device)
        z_gp1_flag = torch.equal(torch.sum(torch.abs(z_gp1_tilde.F), dim=1),
                                 torch.zeros(torch.sum(torch.abs(z_gp1_tilde.F), dim=1).shape[0]).cuda())
        prior_gp1 = entropy_model_group1.hyper_decoder(z_gp1_tilde)  # extimate prior from z

        # encode z
        z_gp1_sort = sort_sparse_tensor(z_gp1)
        z_strings, z_min_v, z_max_v = entropy_model_group1.entropy_bottleneck.compress(z_gp1_sort.F)
        z_gp1_shape = z_gp1_sort.F.shape
        if z_gp1_flag == True:
            # print('skip z coding, encoding zeros, send one bit flag')
            z_Bytes_gp1 = 1.0 / 8
        else:
            # print('hyper coding')
            with open(filename + '_z_F1.bin', 'wb') as fout:
                fout.write(z_strings)
            with open(filename + '_z_H1.bin', 'wb') as fout:
                fout.write(np.array(z_gp1_shape, dtype=np.int32).tobytes())
                fout.write(np.array(len(z_min_v), dtype=np.int8).tobytes())
                fout.write(np.array(z_min_v, dtype=np.float32).tobytes())
                fout.write(np.array(z_max_v, dtype=np.float32).tobytes())
            z_Bytes_gp1 = os.path.getsize(filename + '_z_F1.bin') + os.path.getsize(filename + '_z_H1.bin')

        # encode group1
        # channel_wise
        prior_gp1 = sort_sparse_tensor(prior_gp1)
        group = sort_sparse_tensor(group)
        y_slices_F = torch.split(group.F, entropy_model_group1.num_slices, dim=-1)
        y_hat_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            if slice_index == 0:
                prior_slice = entropy_model_group1.cc_transforms[slice_index](prior_gp1)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)

                y_strings, y_min_v, y_max_v = entropy_model_group1.conditional_entropy_models[slice_index].compress(
                    y_slice_F, loc, scale)

                y_hat_slice_F = entropy_model_group1.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                                       mode="symbols")
                y_slice_F = y_slice_F.cuda()

                y_shape = y_slice_F.shape
                with open(filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(y_strings)
                with open(filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(np.array(y_shape, dtype=np.int32).tobytes())
                    fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
                y_Bytes_gp1 = os.path.getsize(
                    filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin') + os.path.getsize(
                    filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin')
                y_hat_slices.append(y_hat_slice_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                prior_support = ME.SparseTensor(features=torch.cat([prior_gp1.F, y_slice_support], dim=1),
                                                coordinate_map_key=prior_gp1.coordinate_map_key,
                                                coordinate_manager=prior_gp1.coordinate_manager,
                                                device=prior_gp1.device)

                prior_slice = entropy_model_group1.cc_transforms[slice_index](prior_support)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)

                y_strings, y_min_v, y_max_v = entropy_model_group1.conditional_entropy_models[slice_index].compress(
                    y_slice_F, loc, scale)
                y_hat_slice_F = entropy_model_group1.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                                       mode="symbols")

                y_shape = y_slice_F.shape
                with open(filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(y_strings)
                with open(filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(np.array(y_shape, dtype=np.int32).tobytes())
                    fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
                y_Bytes_gp1 += os.path.getsize(
                    filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin') + os.path.getsize(
                    filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin')
                y_hat_slices.append(y_hat_slice_F)
        y_gp1_hat = ME.SparseTensor(features=torch.cat(y_hat_slices, dim=1), coordinates=group.C, device=group.device)

        # group2 hyper
        z_gp2 = entropy_model_group2.hyper_encoder(group2)
        z_gp2_F = entropy_model_group2.entropy_bottleneck._quantize(z_gp2.F, mode="symbols")
        z_gp2_tilde = ME.SparseTensor(features=z_gp2_F, coordinate_map_key=z_gp2.coordinate_map_key,
                                      coordinate_manager=z_gp2.coordinate_manager, device=z_gp2.device)
        z_gp2_flag = torch.equal(torch.sum(torch.abs(z_gp2_tilde.F), dim=1),
                                 torch.zeros(torch.sum(torch.abs(z_gp2_tilde.F), dim=1).shape[0]).cuda())
        prior_gp2 = entropy_model_group2.hyper_decoder(z_gp2_tilde)  # extimate prior from z
        # encode z
        z_strings, z_min_v, z_max_v = entropy_model_group2.entropy_bottleneck.compress(z_gp2.F)
        z_gp2_shape = z_gp2.F.shape
        if z_gp2_flag:
            # print('skip z coding, encoding zeros, send one bit flag')
            z_Bytes_gp2 = 1.0 / 8
        else:
            # print('hyper coding')
            with open(filename + '_z_F2.bin', 'wb') as fout:
                fout.write(z_strings)
            with open(filename + '_z_H2.bin', 'wb') as fout:
                fout.write(np.array(z_gp2_shape, dtype=np.int32).tobytes())
                fout.write(np.array(len(z_min_v), dtype=np.int8).tobytes())
                fout.write(np.array(z_min_v, dtype=np.float32).tobytes())
                fout.write(np.array(z_max_v, dtype=np.float32).tobytes())
            z_Bytes_gp2 = os.path.getsize(filename + '_z_F2.bin') + os.path.getsize(filename + '_z_H2.bin')

        # encoder gp2
        prior_gp2 = sort_sparse_tensor(prior_gp2)  # resort
        group2 = sort_sparse_tensor(group2)
        y_slices_F = torch.split(group2.F, entropy_model_group2.num_slices, dim=-1)
        y_hat_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            y_slice = ME.SparseTensor(features=y_slice_F, coordinates=group2.C, device=group2.device)
            if slice_index == 0:
                loc, scale = entropy_model_group2.context_models[slice_index](y_gp1_hat, y_slice, prior_gp2,
                                                                              y_slice_support=torch.tensor([],
                                                                                                           device='cuda'))
                scale = torch.clamp(scale, min=1e-8)
                y_strings, y_min_v, y_max_v = entropy_model_group2.conditional_entropy_models[slice_index].compress(
                    y_slice_F, loc, scale)
                y_hat_slice_F = entropy_model_group2.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                                       mode="symbols")
                y_hat_slices.append(y_hat_slice_F)
                with open(filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(y_strings)
                with open(filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(np.array(y_shape, dtype=np.int32).tobytes())
                    fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
                y_Bytes_gp2 = os.path.getsize(
                    filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin') + os.path.getsize(
                    filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin')
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                loc, scale = entropy_model_group2.context_models[slice_index](y_gp1_hat, y_slice, prior_gp2,
                                                                              y_slice_support=y_slice_support)
                scale = torch.clamp(scale, min=1e-8)
                y_strings, y_min_v, y_max_v = entropy_model_group2.conditional_entropy_models[slice_index].compress(
                    y_slice_F, loc, scale)
                y_hat_slice_F = entropy_model_group2.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                                       mode="symbols")
                y_hat_slices.append(y_hat_slice_F)
                with open(filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(y_strings)
                with open(filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(np.array(y_shape, dtype=np.int32).tobytes())
                    # fout.write(np.array(len(y_min_v), dtype=np.int8).tobytes())
                    fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
                y_Bytes_gp2 += os.path.getsize(
                    filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin') + os.path.getsize(
                    filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin')

        y_Bytes = y_Bytes_gp1 + y_Bytes_gp2
        z_Bytes = z_Bytes_gp1 + z_Bytes_gp2

        return y_Bytes, z_Bytes, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, nums_list

    @torch.no_grad()
    def decode(self, qs, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, nums_list, down_filename, filename, device='cuda'):
        lmb_input = self.lambda_transform_geo(qs)

        ################################################ Entropy Model ################################################
        entropy_model_list1 = [self.entropy_bottleneck_1.entropy_bottleneck_0,
                               self.entropy_bottleneck_1.entropy_bottleneck_1,
                               self.entropy_bottleneck_1.entropy_bottleneck_2,
                               self.entropy_bottleneck_1.entropy_bottleneck_3]
        entropy_model_list2 = [self.entropy_bottleneck_2.entropy_bottleneck_0,
                               self.entropy_bottleneck_2.entropy_bottleneck_1,
                               self.entropy_bottleneck_2.entropy_bottleneck_2,
                               self.entropy_bottleneck_2.entropy_bottleneck_3]
        entropy_model_group1 = entropy_model_list1[index_gp1]
        entropy_model_group2 = entropy_model_list2[index_gp2]

        # decode coords
        y = load_sparse_tensor(down_filename, order='gbr')
        y = ME.SparseTensor(features=torch.ones(len(y)).unsqueeze(1), coordinates=y.C * 4,
                            tensor_stride=1, device=device)

        downsamplerB = torch.nn.Sequential(*[ME.MinkowskiMaxPooling(
            kernel_size=2, stride=2, dimension=3)] * 2).to(device)

        group, group2 = self.build_group(y)
        z_gp1 = downsamplerB(group)
        z_gp2 = downsamplerB(group2)
        # decode z_gp1
        if z_gp1_flag:
            # print('decoding zeros')
            z_F = torch.zeros([z_gp1.shape[0], entropy_model_group1.channel]).cuda()
        else:
            # print('decoding z')
            with open(filename + '_z_F1.bin', 'rb') as fin:
                z_strings = fin.read()
            with open(filename + '_z_H1.bin', 'rb') as fin:
                z_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                z_min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
                z_max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
            z_F = entropy_model_group1.entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape,
                                                                     channels=z_shape[-1])

        z_gp1_sort = sort_sparse_tensor(z_gp1)
        z = ME.SparseTensor(features=z_F,
                            coordinate_map_key=z_gp1_sort.coordinate_map_key,
                            coordinate_manager=z_gp1_sort.coordinate_manager,
                            device=z_gp1_sort.device)
        z_new = self.sort(z, z_gp1.C)
        z = ME.SparseTensor(features=z_new.F,
                            coordinate_map_key=z_gp1.coordinate_map_key,
                            coordinate_manager=z_gp1.coordinate_manager,
                            device=z_gp1.device)
        # decode y
        prior_gp1 = entropy_model_group1.hyper_decoder(z)
        prior_gp1 = sort_sparse_tensor(prior_gp1)
        group = sort_sparse_tensor(group)
        y_slices_F = torch.split(torch.zeros([group.shape[0], entropy_model_group1.channel]),
                                 entropy_model_group1.num_slices, dim=-1)
        y_hat_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            if slice_index == 0:
                prior_slice = entropy_model_group1.cc_transforms[slice_index](prior_gp1)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)

                with open(filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_strings = fin.read()
                with open(filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                    y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]

                y_F = entropy_model_group1.conditional_entropy_models[slice_index].decompress(y_strings, loc, scale,
                                                                                              y_min_v, y_max_v, y_shape,
                                                                                              channels=y_shape[-1])
                y_F = y_F.cuda()

                y_hat_slices.append(y_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                prior_support = ME.SparseTensor(features=torch.cat([prior_gp1.F, y_slice_support], dim=1),
                                                coordinate_map_key=prior_gp1.coordinate_map_key,
                                                coordinate_manager=prior_gp1.coordinate_manager,
                                                device=prior_gp1.device)

                prior_slice = entropy_model_group1.cc_transforms[slice_index](prior_support)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)
                with open(filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_strings = fin.read()
                with open(filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                    y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]

                y_F = entropy_model_group1.conditional_entropy_models[slice_index].decompress(y_strings, loc, scale,
                                                                                              y_min_v, y_max_v, y_shape,
                                                                                              channels=y_shape[-1])
                y_F = y_F.cuda()
                y_hat_slices.append(y_F)
        y_gp1_hat = ME.SparseTensor(features=torch.cat(y_hat_slices, dim=1), coordinates=group.C, device=group.device)

        # decode z_gp2
        if z_gp2_flag:
            # print('decoding zeros')
            z_F = torch.zeros([z_gp2.shape[0], entropy_model_group2.channel]).cuda()
        else:
            # print('decoding z')
            with open(filename + '_z_F2.bin', 'rb') as fin:
                z_strings = fin.read()
            with open(filename + '_z_H2.bin', 'rb') as fin:
                z_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                z_min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
                z_max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
            z_F = entropy_model_group2.entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape,
                                                                     channels=z_shape[-1])
        z = ME.SparseTensor(features=z_F,
                            coordinate_map_key=z_gp2.coordinate_map_key,
                            coordinate_manager=z_gp2.coordinate_manager,
                            device=z_gp2.device)
        # decode y
        prior_gp2 = entropy_model_group2.hyper_decoder(z)

        group2 = sort_sparse_tensor(group2)  # resort
        prior_gp2 = sort_sparse_tensor(prior_gp2)  # resort

        y_slices_F = torch.split(torch.zeros([group2.shape[0], entropy_model_group2.channel]),
                                 entropy_model_group2.num_slices, dim=-1)
        y_hat_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            y_slice = ME.SparseTensor(features=y_slice_F, coordinates=group2.C, device=group2.device)
            if slice_index == 0:
                loc, scale = entropy_model_group2.context_models[slice_index](y_gp1_hat, y_slice, prior_gp2,
                                                                              y_slice_support=torch.tensor([],
                                                                                                           device='cuda'))
                scale = torch.clamp(scale, min=1e-8)
                with open(filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_strings = fin.read()
                with open(filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                    y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]

                y_F = entropy_model_group2.conditional_entropy_models[slice_index].decompress(y_strings, loc, scale,
                                                                                              y_min_v, y_max_v, y_shape,
                                                                                              channels=y_shape[-1])
                y_F = y_F.cuda()
                y_hat_slices.append(y_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                loc, scale = entropy_model_group2.context_models[slice_index](y_gp1_hat, y_slice, prior_gp2,
                                                                              y_slice_support=y_slice_support)
                scale = torch.clamp(scale, min=1e-8)
                with open(filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_strings = fin.read()
                with open(filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                    y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                y_F = entropy_model_group2.conditional_entropy_models[slice_index].decompress(y_strings, loc, scale,
                                                                                              y_min_v, y_max_v, y_shape,
                                                                                              channels=y_shape[-1])
                y_F = y_F.cuda()
                y_hat_slices.append(y_F)

        y_gp2_hat = ME.SparseTensor(features=torch.cat(y_hat_slices, dim=1), coordinates=group2.C, device=group2.device)
        y_gp_C = torch.cat([y_gp1_hat.C * 4, y_gp2_hat.C * 4], dim=0)
        y_gp_F = torch.cat([y_gp1_hat.F, y_gp2_hat.F], dim=0)
        y_gp_new = self.build_sparsetensor(y_gp_C, y_gp_F)
        y_hat = self.sort(y_gp_new, y.C)
        y_tilde = ME.SparseTensor(features=y_hat.F, coordinates=y.C,
                                  tensor_stride=4, device=device)

        ################################################ Decoder ################################################
        down = ME.SparseTensor(
            features=(torch.ones(len(y_tilde)).unsqueeze(1)),
            coordinates=y_tilde.C,
            tensor_stride=4,
            device=y_tilde.device)
        down = self.extractor(down, lmb_input)

        y_tilde = self.channel_transform_dec(y_tilde)
        _, out, _ = self.decoder_G(y_tilde, down, lmb_input, nums_list, [None]*3,
                                                            training=False)

        return out

class PCACModel(torch.nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        # transform
        self.lambda_transform_attr = LambdaTransform(channels)
        self.extractor = SIA(inchannels=3)
        self.encoder_A = Encoder_A(channels=channels)
        self.decoder_A = Decoder_A(channels=channels)
        self.channel_transform_enc = ChannelTransform(channels, channels//4)
        self.channel_transform_dec = ChannelTransform(channels//4, channels)

        # entropy model
        self.entropy_bottleneck_1 = HyperChannelwiseEntropyModelGroup(32, [16, 16])
        self.entropy_bottleneck_2 = HyperCheckboardChannelwiseEntropyModelGroup(32, [16, 16])

        # tools
        self.downsampler = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        self.sort = ME.MinkowskiMaxPooling(kernel_size=1, stride=1, dimension=3)

    def sort_sparse_tensor(self, sparse_tensor):
        """ Sort points in sparse tensor according to their coordinates.
        """
        coords_new = sparse_tensor.C[:,1:] // 4

        mask = (coords_new[:,2] % 2 == 0) & (coords_new[:,1] % 2 == 0) & (coords_new[:,0] % 2 == 0)
        mask2 = (coords_new[:,2] % 2 == 0) & (coords_new[:,1] % 2 == 1) & (coords_new[:,0] % 2 == 1)
        mask3 = (coords_new[:,2] % 2 == 1) & (coords_new[:,1] % 2 == 1) & (coords_new[:,0] % 2 == 0)
        mask4 = (coords_new[:,2] % 2 == 1) & (coords_new[:,1] % 2 == 0) & (coords_new[:,0] % 2 == 1)

        mask_odd = mask | mask2 | mask3 | mask4

        return mask_odd

    def build_sparsetensor(self, coords, feats):
        sparsetensor = ME.SparseTensor(features=feats, coordinates=coords, device='cuda')

        return sparsetensor
    def build_group(self, y):
        indices_sort = self.sort_sparse_tensor(y)
        y_coords_group_list = []
        y_feats_group_list = []
        y_coords_group2_list = []
        y_feats_group2_list = []

        y_coords_group_list.append(y.C[indices_sort == True] // 4)
        y_feats_group_list.append(y.F[indices_sort == True])
        y_coords_group2_list.append(y.C[indices_sort == False] // 4)
        y_feats_group2_list.append(y.F[indices_sort == False])

        y_coords_group = torch.cat(y_coords_group_list, dim=0)
        y_feats_group = torch.cat(y_feats_group_list, dim=0)
        y_coords_group2 = torch.cat(y_coords_group2_list, dim=0)
        y_feats_group2 = torch.cat(y_feats_group2_list, dim=0)

        group = self.build_sparsetensor(y_coords_group, y_feats_group)
        group2 = self.build_sparsetensor(y_coords_group2, y_feats_group2)

        return group, group2

    def quantize(self, inputs):
        from models.entropy_model import RoundNoGradient

        return RoundNoGradient.apply(inputs)

    def inv_group(self, y, group1_tilde, group2_tilde, y_likelihood_1, y_likelihood_2, z_likelihood_1, z_likelihood_2):
        y_likelihoods = []
        z_likelihoods = []

        y_tilde12_C = torch.cat([group1_tilde.C, group2_tilde.C], dim=0)
        y_tilde12_F = torch.cat([group1_tilde.F, group2_tilde.F], dim=0)

        y_tilde12 = ME.SparseTensor(features=y_tilde12_F, coordinates=y_tilde12_C, device='cuda')

        y_likelihoods.append(y_likelihood_1)
        y_likelihoods.append(y_likelihood_2)
        z_likelihoods.append(z_likelihood_1)
        z_likelihoods.append(z_likelihood_2)
        z_likelihood = torch.cat(z_likelihoods, dim=0)
        y_likelihood = torch.cat(y_likelihoods, dim=0)

        y_gp_C = torch.cat([y_tilde12.C * 4], dim=0)
        y_gp_F = torch.cat([y_tilde12.F], dim=0)
        y_gp_new = self.build_sparsetensor(y_gp_C, y_gp_F)

        y_hat = self.sort(y_gp_new, y.C)
        y_tilde = ME.SparseTensor(features=y_hat.F, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)

        return y_tilde, y_likelihood, z_likelihood

    def forward_attr(self, current_x, current_down, qs, training, eval=False):

        # down
        true_down = self.downsampler(self.downsampler(current_x))
        current_x_down = self.sort(current_down, true_down.C)
        current_x_down = ME.SparseTensor(
            features=current_x_down.F,
            coordinate_map_key=true_down.coordinate_map_key,
            coordinate_manager=true_down.coordinate_manager,
            device=true_down.device)

        lmb_input = self.lambda_transform_attr(qs)

        down = self.extractor(current_x_down, lmb_input)

        y = self.encoder_A(current_x, lmb_input)

        y = self.channel_transform_enc(y)

        # Quantizer & Entropy Model
        groupA, groupB = self.build_group(y)
        y_tilde1, _, y_likelihood_1, z_likelihood_1, _ = self.entropy_bottleneck_1(groupA, qs, training, eval)
        y_tilde2, _, y_likelihood_2, z_likelihood_2, _ = self.entropy_bottleneck_2(groupB, y_tilde1, qs, training, eval)

        y_tilde, y_likelihood, z_likelihood = self.inv_group(y, y_tilde1,
                                                        y_tilde2, y_likelihood_1, y_likelihood_2, z_likelihood_1,
                                                        z_likelihood_2)

        y_tilde = self.channel_transform_dec(y_tilde)

        out = self.decoder_A(y_tilde, down, current_x_down, lmb_input)

        return {'likelihood': y_likelihood,
                'z_likelihood': z_likelihood,
                'y_tilde': y_tilde,
                'out': out,
                'x': current_x}

    @torch.no_grad()
    def encode(self, x, qs, down_filename, filename, guided):
        lmb_input = self.lambda_transform_attr(qs)

        ################################################ Encoder ################################################
        x = sort_sparse_tensor(x)

        # Encoder
        y = self.encoder_A(x, lmb_input)
        y = self.channel_transform_enc(y)

        ################################################ Entropy Model ################################################
        group, group2 = self.build_group(y)
        group_tilde1, _, _, _, index_gp1 = self.entropy_bottleneck_1(group, qs, False, True)
        _, _, _, _, index_gp2 = self.entropy_bottleneck_2(group2, group_tilde1, qs, False, True)

        entropy_model_list1 = [self.entropy_bottleneck_1.entropy_bottleneck_0,
                               self.entropy_bottleneck_1.entropy_bottleneck_1,
                               self.entropy_bottleneck_1.entropy_bottleneck_2,
                               self.entropy_bottleneck_1.entropy_bottleneck_3]
        entropy_model_list2 = [self.entropy_bottleneck_2.entropy_bottleneck_0,
                               self.entropy_bottleneck_2.entropy_bottleneck_1,
                               self.entropy_bottleneck_2.entropy_bottleneck_2,
                               self.entropy_bottleneck_2.entropy_bottleneck_3]
        entropy_model_group1 = entropy_model_list1[index_gp1]
        entropy_model_group2 = entropy_model_list2[index_gp2]

        # group1 hyper
        z_gp1 = entropy_model_group1.hyper_encoder(group)

        z_gp1_F = entropy_model_group1.entropy_bottleneck._quantize(z_gp1.F, mode="symbols")
        z_gp1_tilde = ME.SparseTensor(features=z_gp1_F, coordinate_map_key=z_gp1.coordinate_map_key,
                                      coordinate_manager=z_gp1.coordinate_manager, device=z_gp1.device)
        z_gp1_flag = torch.equal(torch.sum(torch.abs(z_gp1_tilde.F), dim=1),
                                 torch.zeros(torch.sum(torch.abs(z_gp1_tilde.F), dim=1).shape[0]).cuda())

        prior_gp1 = entropy_model_group1.hyper_decoder(z_gp1_tilde)  # extimate prior from z

        # encode z
        z_gp1_sort = sort_sparse_tensor(z_gp1)
        z_strings, z_min_v, z_max_v = entropy_model_group1.entropy_bottleneck.compress(z_gp1_sort.F)
        z_gp1_shape = z_gp1_sort.F.shape
        if z_gp1_flag == True:
            # print('skip z coding, encoding zeros, send one bit flag')
            z_Bytes_gp1 = 1.0 / 8
        else:
            # print('hyper coding')
            with open(filename + '_z_F1.bin', 'wb') as fout:
                fout.write(z_strings)
            with open(filename + '_z_H1.bin', 'wb') as fout:
                fout.write(np.array(z_gp1_shape, dtype=np.int32).tobytes())
                fout.write(np.array(len(z_min_v), dtype=np.int8).tobytes())
                fout.write(np.array(z_min_v, dtype=np.float32).tobytes())
                fout.write(np.array(z_max_v, dtype=np.float32).tobytes())
            z_Bytes_gp1 = os.path.getsize(filename + '_z_F1.bin') + os.path.getsize(filename + '_z_H1.bin')

        # encode group1
        # channel_wise
        prior_gp1 = sort_sparse_tensor(prior_gp1)
        group = sort_sparse_tensor(group)
        y_slices_F = torch.split(group.F, entropy_model_group1.num_slices, dim=-1)
        y_hat_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            if slice_index == 0:
                prior_slice = entropy_model_group1.cc_transforms[slice_index](prior_gp1)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)

                y_strings, y_min_v, y_max_v = entropy_model_group1.conditional_entropy_models[slice_index].compress(
                    y_slice_F, loc, scale)

                y_hat_slice_F = entropy_model_group1.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                                       mode="symbols")
                y_slice_F = y_slice_F.cuda()

                y_shape = y_slice_F.shape
                with open(filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(y_strings)
                with open(filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(np.array(y_shape, dtype=np.int32).tobytes())
                    fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
                y_Bytes_gp1 = os.path.getsize(
                    filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin') + os.path.getsize(
                    filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin')
                y_hat_slices.append(y_hat_slice_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                prior_support = ME.SparseTensor(features=torch.cat([prior_gp1.F, y_slice_support], dim=1),
                                                coordinate_map_key=prior_gp1.coordinate_map_key,
                                                coordinate_manager=prior_gp1.coordinate_manager,
                                                device=prior_gp1.device)

                prior_slice = entropy_model_group1.cc_transforms[slice_index](prior_support)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)

                y_strings, y_min_v, y_max_v = entropy_model_group1.conditional_entropy_models[slice_index].compress(
                    y_slice_F, loc, scale)
                y_hat_slice_F = entropy_model_group1.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                                       mode="symbols")

                y_shape = y_slice_F.shape
                with open(filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(y_strings)
                with open(filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(np.array(y_shape, dtype=np.int32).tobytes())
                    fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
                y_Bytes_gp1 += os.path.getsize(
                    filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin') + os.path.getsize(
                    filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin')
                y_hat_slices.append(y_hat_slice_F)
        y_gp1_hat = ME.SparseTensor(features=torch.cat(y_hat_slices, dim=1), coordinates=group.C, device=group.device)

        # group2 hyper
        z_gp2 = entropy_model_group2.hyper_encoder(group2)
        z_gp2_F = entropy_model_group2.entropy_bottleneck._quantize(z_gp2.F, mode="symbols")
        z_gp2_tilde = ME.SparseTensor(features=z_gp2_F, coordinate_map_key=z_gp2.coordinate_map_key,
                                      coordinate_manager=z_gp2.coordinate_manager, device=z_gp2.device)
        z_gp2_flag = torch.equal(torch.sum(torch.abs(z_gp2_tilde.F), dim=1),
                                 torch.zeros(torch.sum(torch.abs(z_gp2_tilde.F), dim=1).shape[0]).cuda())
        prior_gp2 = entropy_model_group2.hyper_decoder(z_gp2_tilde)  # extimate prior from z
        # encode z
        z_strings, z_min_v, z_max_v = entropy_model_group2.entropy_bottleneck.compress(z_gp2.F)
        z_gp2_shape = z_gp2.F.shape
        if z_gp2_flag:
            # print('skip z coding, encoding zeros, send one bit flag')
            z_Bytes_gp2 = 1.0 / 8
        else:
            # print('hyper coding')
            with open(filename + '_z_F2.bin', 'wb') as fout:
                fout.write(z_strings)
            with open(filename + '_z_H2.bin', 'wb') as fout:
                fout.write(np.array(z_gp2_shape, dtype=np.int32).tobytes())
                fout.write(np.array(len(z_min_v), dtype=np.int8).tobytes())
                fout.write(np.array(z_min_v, dtype=np.float32).tobytes())
                fout.write(np.array(z_max_v, dtype=np.float32).tobytes())
            z_Bytes_gp2 = os.path.getsize(filename + '_z_F2.bin') + os.path.getsize(filename + '_z_H2.bin')

        # encoder gp2
        prior_gp2 = sort_sparse_tensor(prior_gp2)  # resort
        group2 = sort_sparse_tensor(group2)
        y_slices_F = torch.split(group2.F, entropy_model_group2.num_slices, dim=-1)
        y_hat_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            y_slice = ME.SparseTensor(features=y_slice_F, coordinates=group2.C, device=group2.device)
            if slice_index == 0:
                loc, scale = entropy_model_group2.context_models[slice_index](y_gp1_hat, y_slice, prior_gp2,
                                                                              y_slice_support=torch.tensor([],
                                                                                                           device='cuda'))
                scale = torch.clamp(scale, min=1e-8)
                y_strings, y_min_v, y_max_v = entropy_model_group2.conditional_entropy_models[slice_index].compress(
                    y_slice_F, loc, scale)
                y_hat_slice_F = entropy_model_group2.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                                       mode="symbols")
                y_hat_slices.append(y_hat_slice_F)
                with open(filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(y_strings)
                with open(filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(np.array(y_shape, dtype=np.int32).tobytes())
                    fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
                y_Bytes_gp2 = os.path.getsize(
                    filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin') + os.path.getsize(
                    filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin')
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                loc, scale = entropy_model_group2.context_models[slice_index](y_gp1_hat, y_slice, prior_gp2,
                                                                              y_slice_support=y_slice_support)
                scale = torch.clamp(scale, min=1e-8)
                y_strings, y_min_v, y_max_v = entropy_model_group2.conditional_entropy_models[slice_index].compress(
                    y_slice_F, loc, scale)
                y_hat_slice_F = entropy_model_group2.conditional_entropy_models[slice_index]._quantize(y_slice_F,
                                                                                                       mode="symbols")
                y_hat_slices.append(y_hat_slice_F)
                with open(filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(y_strings)
                with open(filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin', 'wb') as fout:
                    fout.write(np.array(y_shape, dtype=np.int32).tobytes())
                    fout.write(np.array(y_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(y_max_v, dtype=np.float32).tobytes())
                y_Bytes_gp2 += os.path.getsize(
                    filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin') + os.path.getsize(
                    filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin')

        y_gp2_hat = ME.SparseTensor(features=torch.cat(y_hat_slices, dim=1), coordinates=group2.C, device=group2.device)

        y_gp_C = torch.cat([y_gp1_hat.C * 4, y_gp2_hat.C * 4], dim=0)
        y_gp_F = torch.cat([y_gp1_hat.F, y_gp2_hat.F], dim=0)
        y_gp_new = self.build_sparsetensor(y_gp_C, y_gp_F)
        y_hat = self.sort(y_gp_new, y.C)
        y_tilde = ME.SparseTensor(features=y_hat.F, coordinate_map_key=y.coordinate_map_key,
                                  coordinate_manager=y.coordinate_manager, device=y.device)

        y_Bytes = y_Bytes_gp1 + y_Bytes_gp2
        z_Bytes = z_Bytes_gp1 + z_Bytes_gp2

        ################################################ Guided ################################################
        # base layer
        downsampler = torch.nn.Sequential(*[ME.MinkowskiAvgPooling(
            kernel_size=2, stride=2, dimension=3)] * 2)
        true_down = downsampler(x)

        x_down = load_sparse_tensor(down_filename, order='gbr')
        x_down = ME.SparseTensor(features=rgb2yuv(x_down.F.clone()), coordinates=x_down.C, device=y.device)
        true_down = ME.SparseTensor(features=true_down.F, coordinates=true_down.C//4, device=y.device)

        x_down = knn_interpolation(x_down, true_down, k=1)
        x_down = ME.SparseTensor(
            features=x_down,
            coordinate_map_key=true_down.coordinate_map_key,
            coordinate_manager=true_down.coordinate_manager,
            device=true_down.device)

        A_offset_down_u, A_offset_down_v, x_down = guided.encode(x_down, true_down)

        A_offset = torch.cat([A_offset_down_u, A_offset_down_v], dim=0)

        x_down = ME.SparseTensor(features=x_down.F, coordinates=x_down.C * 4, device=y.device)
        x_down = self.sort(x_down, y_tilde.C)
        x_down = ME.SparseTensor(
            features=x_down.F,
            coordinate_map_key=y_tilde.coordinate_map_key,
            coordinate_manager=y_tilde.coordinate_manager,
            device=y_tilde.device)

        # enhanment layer
        down = self.extractor(x_down, lmb_input)
        y_tilde = self.channel_transform_dec(y_tilde)

        out = self.decoder_A(y_tilde, down, x_down, lmb_input)

        A_offset_u, A_offset_v, out = guided.encode(out, x)

        A_offset = torch.cat([A_offset, A_offset_u], dim=0)
        A_offset = torch.cat([A_offset, A_offset_v], dim=0)
        A_offset = A_offset.permute(1, 0)

        from models.entropy_model import EntropyBottleneck
        entropy_bottleneck = EntropyBottleneck(channels=12)
        strings, min_v, max_v = entropy_bottleneck.compress(A_offset.cuda())
        shape = A_offset.shape
        with open(filename + '_offset_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(filename + '_offset_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
            fout.write(np.array(min_v, dtype=np.float32).tobytes())
            fout.write(np.array(max_v, dtype=np.float32).tobytes())
        Guided_bytes = os.path.getsize(filename + '_offset_F.bin') + os.path.getsize(filename + '_offset_H.bin')

        return y_Bytes, z_Bytes, Guided_bytes, \
            index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, out

    @torch.no_grad()
    def decode(self, x, qs, index_gp1, index_gp2, z_gp1_flag, z_gp2_flag, down_filename, filename, guided, device='cuda'):

        lmb_input = self.lambda_transform_attr(qs)

        x = sort_sparse_tensor(x)
        x = ME.SparseTensor(
            features=(torch.ones(len(x)).unsqueeze(1)),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)


        ################################################ Entropy Model ################################################
        entropy_model_list1 = [self.entropy_bottleneck_1.entropy_bottleneck_0,
                               self.entropy_bottleneck_1.entropy_bottleneck_1,
                               self.entropy_bottleneck_1.entropy_bottleneck_2,
                               self.entropy_bottleneck_1.entropy_bottleneck_3]
        entropy_model_list2 = [self.entropy_bottleneck_2.entropy_bottleneck_0,
                               self.entropy_bottleneck_2.entropy_bottleneck_1,
                               self.entropy_bottleneck_2.entropy_bottleneck_2,
                               self.entropy_bottleneck_2.entropy_bottleneck_3]
        entropy_model_group1 = entropy_model_list1[index_gp1]
        entropy_model_group2 = entropy_model_list2[index_gp2]

        # y & z coords
        downsamplerA = torch.nn.Sequential(*[ME.MinkowskiMaxPooling(
            kernel_size=2, stride=2, dimension=3)] * 2).to(device)
        downsamplerB = torch.nn.Sequential(*[ME.MinkowskiMaxPooling(
            kernel_size=2, stride=2, dimension=3)] * 2).to(device)

        y = downsamplerA(x)
        group, group2 = self.build_group(y)
        z_gp1 = downsamplerB(group)
        z_gp2 = downsamplerB(group2)

        # decode z_gp1
        if z_gp1_flag:
            # print('decoding zeros')
            z_F = torch.zeros([z_gp1.shape[0], entropy_model_group1.channel]).cuda()
        else:
            # print('decoding z')
            with open(filename + '_z_F1.bin', 'rb') as fin:
                z_strings = fin.read()
            with open(filename + '_z_H1.bin', 'rb') as fin:
                z_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                z_min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
                z_max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
            z_F = entropy_model_group1.entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape,
                                                                     channels=z_shape[-1])

        z_gp1_sort = sort_sparse_tensor(z_gp1)
        z = ME.SparseTensor(features=z_F,
                            coordinate_map_key=z_gp1_sort.coordinate_map_key,
                            coordinate_manager=z_gp1_sort.coordinate_manager,
                            device=z_gp1_sort.device)
        z_new = self.sort(z, z_gp1.C)
        z = ME.SparseTensor(features=z_new.F,
                            coordinate_map_key=z_gp1.coordinate_map_key,
                            coordinate_manager=z_gp1.coordinate_manager,
                            device=z_gp1.device)
        # decode y
        prior_gp1 = entropy_model_group1.hyper_decoder(z)
        prior_gp1 = sort_sparse_tensor(prior_gp1)
        group = sort_sparse_tensor(group)
        y_slices_F = torch.split(torch.zeros([group.shape[0], entropy_model_group1.channel]),
                                 entropy_model_group1.num_slices, dim=-1)
        y_hat_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            if slice_index == 0:
                prior_slice = entropy_model_group1.cc_transforms[slice_index](prior_gp1)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)

                with open(filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_strings = fin.read()
                with open(filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                    y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]

                y_F = entropy_model_group1.conditional_entropy_models[slice_index].decompress(y_strings, loc, scale,
                                                                                              y_min_v, y_max_v, y_shape,
                                                                                              channels=y_shape[-1])
                y_F = y_F.cuda()

                y_hat_slices.append(y_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                prior_support = ME.SparseTensor(features=torch.cat([prior_gp1.F, y_slice_support], dim=1),
                                                coordinate_map_key=prior_gp1.coordinate_map_key,
                                                coordinate_manager=prior_gp1.coordinate_manager,
                                                device=prior_gp1.device)

                prior_slice = entropy_model_group1.cc_transforms[slice_index](prior_support)
                loc = prior_slice.F[:, :prior_slice.F.shape[-1] // 2]
                scale = prior_slice.F[:, prior_slice.F.shape[-1] // 2:].abs()
                scale = torch.clamp(scale, min=1e-8)
                with open(filename + '_y_F_gp1' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_strings = fin.read()
                with open(filename + '_y_H_gp1' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                    y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]

                y_F = entropy_model_group1.conditional_entropy_models[slice_index].decompress(y_strings, loc, scale,
                                                                                              y_min_v, y_max_v, y_shape,
                                                                                              channels=y_shape[-1])
                y_F = y_F.cuda()
                y_hat_slices.append(y_F)
        y_gp1_hat = ME.SparseTensor(features=torch.cat(y_hat_slices, dim=1), coordinates=group.C, device=group.device)

        # decode z_gp2
        if z_gp2_flag:
            # print('decoding zeros')
            z_F = torch.zeros([z_gp2.shape[0], entropy_model_group2.channel]).cuda()
        else:
            # print('decoding z')
            with open(filename + '_z_F2.bin', 'rb') as fin:
                z_strings = fin.read()
            with open(filename + '_z_H2.bin', 'rb') as fin:
                z_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                z_min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
                z_max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
            z_F = entropy_model_group2.entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape,
                                                                     channels=z_shape[-1])
        z = ME.SparseTensor(features=z_F,
                            coordinate_map_key=z_gp2.coordinate_map_key,
                            coordinate_manager=z_gp2.coordinate_manager,
                            device=z_gp2.device)
        # decode y
        prior_gp2 = entropy_model_group2.hyper_decoder(z)

        group2 = sort_sparse_tensor(group2)  # resort
        prior_gp2 = sort_sparse_tensor(prior_gp2)  # resort

        y_slices_F = torch.split(torch.zeros([group2.shape[0], entropy_model_group2.channel]),
                                 entropy_model_group2.num_slices, dim=-1)
        y_hat_slices = []
        for slice_index, y_slice_F in enumerate(y_slices_F):
            y_slice = ME.SparseTensor(features=y_slice_F, coordinates=group2.C, device=group2.device)
            if slice_index == 0:
                loc, scale = entropy_model_group2.context_models[slice_index](y_gp1_hat, y_slice, prior_gp2,
                                                                              y_slice_support=torch.tensor([],
                                                                                                           device='cuda'))
                scale = torch.clamp(scale, min=1e-8)
                with open(filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_strings = fin.read()
                with open(filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                    y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]

                y_F = entropy_model_group2.conditional_entropy_models[slice_index].decompress(y_strings, loc, scale,
                                                                                              y_min_v, y_max_v, y_shape,
                                                                                              channels=y_shape[-1])
                y_F = y_F.cuda()
                y_hat_slices.append(y_F)
            else:
                y_slice_support = torch.cat(y_hat_slices, dim=1)
                loc, scale = entropy_model_group2.context_models[slice_index](y_gp1_hat, y_slice, prior_gp2,
                                                                              y_slice_support=y_slice_support)
                scale = torch.clamp(scale, min=1e-8)
                with open(filename + '_y_F_gp2' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_strings = fin.read()
                with open(filename + '_y_H_gp2' + '_' + str(slice_index) + '.bin', 'rb') as fin:
                    y_shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
                    y_min_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                    y_max_v = np.frombuffer(fin.read(4), dtype=np.float32)[0]
                y_F = entropy_model_group2.conditional_entropy_models[slice_index].decompress(y_strings, loc, scale,
                                                                                              y_min_v, y_max_v, y_shape,
                                                                                              channels=y_shape[-1])
                y_F = y_F.cuda()
                y_hat_slices.append(y_F)

        y_gp2_hat = ME.SparseTensor(features=torch.cat(y_hat_slices, dim=1), coordinates=group2.C, device=group2.device)
        y_gp_C = torch.cat([y_gp1_hat.C * 4, y_gp2_hat.C * 4], dim=0)
        y_gp_F = torch.cat([y_gp1_hat.F, y_gp2_hat.F], dim=0)
        y_gp_new = self.build_sparsetensor(y_gp_C, y_gp_F)
        y_hat = self.sort(y_gp_new, y.C)
        y_tilde = ME.SparseTensor(features=y_hat.F, coordinates=y.C,
                                  tensor_stride=4, device=device)

        y_tilde = self.sort(y_tilde, y.C)
        y_tilde = ME.SparseTensor(
            features=y_tilde.F,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)

        from models.entropy_model import EntropyBottleneck
        entropy_bottleneck = EntropyBottleneck(channels=12)
        with open(filename + '_offset_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(filename + '_offset_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4 * 2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
            max_v = np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)
        offset = entropy_bottleneck.decompress(strings, min_v, max_v, shape, channels=12)

        ################################################ Decoder ################################################
        x_down = load_sparse_tensor(down_filename, order='gbr')
        x_down = ME.SparseTensor(features=rgb2yuv(x_down.F), coordinates=x_down.C * 4, device=y.device)
        x_down = knn_interpolation(x_down, y, k=1)
        x_down = ME.SparseTensor(
            features=x_down,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)
        x_down = ME.SparseTensor(features=x_down.F, coordinates=x_down.C / 4, device=y.device)

        # guided
        A_offset_down_u = offset[:, 0:3].cuda().permute(1, 0)
        A_offset_down_v = offset[:, 3:6].cuda().permute(1, 0)

        x_down = guided.decode(x_down, A_offset_down_u, A_offset_down_v)
        x_down = ME.SparseTensor(features=x_down.F, coordinates=x_down.C * 4, device=y.device)
        x_down = self.sort(x_down, y.C)
        x_down = ME.SparseTensor(
            features=x_down.F,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
            device=y.device)

        # Decoder
        down = self.extractor(x_down, lmb_input)

        y_tilde = self.channel_transform_dec(y_tilde)
        out = self.decoder_A(y_tilde, down, x_down, lmb_input)

        # guided
        A_offset_u = offset[:, 6:9].cuda().permute(1, 0)
        A_offset_v = offset[:, 9:12].cuda().permute(1, 0)
        out = guided.decode(out, A_offset_u, A_offset_v)

        return out


if __name__ == '__main__':
    model = PCGCModel(channels=128)
    print('params:',sum(param.numel() for param in model.parameters()))

    model = PCACModel(channels=128)
    print('params:',sum(param.numel() for param in model.parameters()))