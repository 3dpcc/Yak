import torch
import MinkowskiEngine as ME
from models.u_module import Enhancer_u
from models.v_module import Enhancer_v
from data_processing.data_utils import sort_sparse_tensor

class GuidedFilter(torch.nn.Module):
    def __init__(self, ckpt_u=None, ckpt_v=None):
        super().__init__()
        self.scale = 128
        if ckpt_u is not None:
            self.module_u = Enhancer_u()
            self.module_u = self.load_model(ckpt_u, self.module_u)
        if ckpt_v is not None:
            self.module_v = Enhancer_v()
            self.module_v = self.load_model(ckpt_v, self.module_v)

    def load_model(self, ckpt, model):
        ckpt_static = torch.load(ckpt)
        model_dict = model.state_dict()
        pretrained_static_dict = {k: v for k, v in ckpt_static['model'].items() if k in model_dict}
        model_dict.update(pretrained_static_dict)
        model.load_state_dict(model_dict)
        return model

    def encode(self, dec, gt):
        dec = sort_sparse_tensor(dec)
        gt = sort_sparse_tensor(gt)

        A_u = self.module_u.encode(dec, gt)
        A_u = torch.round(A_u * self.scale) + self.scale
        out_u = self.module_u.decode(dec, (A_u - self.scale) / self.scale)

        A_v = self.module_v.encode(dec, gt)
        A_v = torch.round(A_v * self.scale) + self.scale
        out_v = self.module_v.decode(dec, (A_v - self.scale) / self.scale)

        out_F = dec.F.clone()
        out_F[:, 1: 2] = out_u.F[:, 1: 2].clone()
        out_F[:, 2:] = out_v.F[:, 2:].clone()

        out = ME.SparseTensor(features=out_F.clone(), coordinates=dec.C, device=dec.device)

        return A_u, A_v, out

    def decode(self, dec, A_u, A_v):
        dec = sort_sparse_tensor(dec)
        out_u = self.module_u.decode(dec, (A_u - self.scale) / self.scale)
        out_v = self.module_v.decode(dec, (A_v - self.scale) / self.scale)

        out_F = dec.F.clone()
        out_F[:, 1: 2] = out_u.F[:, 1: 2].clone()
        out_F[:, 2:] = out_v.F[:, 2:].clone()

        out = ME.SparseTensor(features=out_F.clone(), coordinates=dec.C, device=dec.device)

        return out


if __name__ == '__main__':
    model = GuidedFilter().to('cuda')
    print('params:',sum(param.numel() for param in model.parameters()))
