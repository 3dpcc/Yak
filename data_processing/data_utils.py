import open3d as o3d
import os
import numpy as np
import pytorch3d.ops
import subprocess
import torch
import MinkowskiEngine as ME

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int32')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 

    return

def write_ply_ascii(filedir, coords, feats):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n', 
                'property uchar red\n','property uchar green\n','property uchar blue\n',])
    f.write('end_header\n')
    coords = coords.astype('int16')
    # coords = coords.astype('float64')
    feats = feats.astype('uint8')
    for xyz, rgb in zip(coords, feats):
        f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
                    str(rgb[0]), ' ', str(rgb[1]), ' ',str(rgb[2]), '\n'])
    f.close() 

    return


def read_ply_ascii(filedir, order='rgb'):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype('int16')
    if data.shape[-1]==6: feats = data[:,3:6].astype('uint8')
    if data.shape[-1]>6: feats = data[:,6:9].astype('uint8')
    if order=='gbr': feats = np.hstack([feats[:,2:3], feats[:,0:2]])

    return coords, feats

def read_ply_ascii_normal(filedir, order='rgb'):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError: continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:,0:3].astype('int16')
    if data.shape[-1]==6: feats = data[:,3:6].astype('uint8')
    if data.shape[-1]>6:
        feats = data[:,6:9].astype('uint8')
        normal = data[:,3:6].astype('float64')
    if order=='gbr': feats = np.hstack([feats[:,2:3], feats[:,0:2]])

    return coords, normal, feats

def write_ply_ascii_normal(filedir, coords, normal, feats):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n',
                  'property float nx\n', 'property float ny\n', 'property float nz\n',
                'property uchar red\n','property uchar green\n','property uchar blue\n',])
    f.write('end_header\n')
    coords = coords.astype('int16')
    normal = normal.astype('float64')
    # coords = coords.astype('float64')
    feats = feats.astype('uint8')
    for xyz, nor, rgb in zip(coords, normal, feats):
        f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
                      str(nor[0]), ' ', str(nor[1]), ' ', str(nor[2]), ' ',
                    str(rgb[0]), ' ', str(rgb[1]), ' ',str(rgb[2]), '\n'])
    f.close()

    return

def write_ply_ascii_with_normal(filedir, coords, feats, knn=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype('int32'))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals = np.array(pcd.normals)
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n',
                  'property float nx\n', 'property float ny\n', 'property float nz\n',
                'property uchar red\n','property uchar green\n','property uchar blue\n',])
    f.write('end_header\n')
    coords = coords.astype('int16')
    feats = feats.astype('uint8')
    for xyz,nor, rgb in zip(coords, normals, feats):
        f.writelines([str(xyz[0]), ' ', str(xyz[1]), ' ',str(xyz[2]), ' ',
                      str(nor[0]), ' ',str(nor[1]), ' ',str(nor[2]), ' ',
                    str(rgb[0]), ' ', str(rgb[1]), ' ',str(rgb[2]), '\n'])
    f.close()

    return

def kdtree_partition(pc, max_num):
    parts = []
    class KD_node:  
        def __init__(self, point=None, LL = None, RR = None):  
            self.point = point  
            self.left = LL  
            self.right = RR
    def createKDTree(root, data):
        if len(data) <= max_num:
            parts.append(data)
            return
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]

        point = data_sorted[int(len(data)/2)]  
        root = KD_node(point)  
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])  
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):]) 
        return root
    init_root = KD_node(None)
    root = createKDTree(init_root, pc)  

    return parts

def sort_points(coords, feats):
    indices_sort = np.argsort(array2vector(coords))

    return coords[indices_sort], feats[indices_sort]


def rgb2yuv(rgb):
    """input: [0,1];    output: [0,1]
    """
    rgb = 255*rgb
    yuv = rgb.clone()
    yuv[:,0] = 0.257*rgb[:,0] + 0.504*rgb[:,1] + 0.098*rgb[:,2] + 16
    yuv[:,1] = -0.148*rgb[:,0] - 0.291*rgb[:,1] + 0.439*rgb[:,2] + 128
    yuv[:,2] = 0.439*rgb[:,0] - 0.368*rgb[:,1] - 0.071*rgb[:,2] + 128
    yuv[:,0] = (yuv[:,0]-16)/(235-16)
    yuv[:,1] = (yuv[:,1]-16)/(240-16)
    yuv[:,2] = (yuv[:,2]-16)/(240-16)
    
    return yuv

def yuv2rgb(yuv):
    """input: [0,1];    output: [0,1]
    """
    yuv[:,0] = (235-16)*yuv[:,0]+16
    yuv[:,1] = (240-16)*yuv[:,1]+16
    yuv[:,2] = (240-16)*yuv[:,2]+16
    rgb = yuv.clone()
    rgb[:,0] = 1.164*(yuv[:,0]-16) + 1.596*(yuv[:,2]-128)
    rgb[:,1] = 1.164*(yuv[:,0]-16) - 0.813*(yuv[:,2]-128) - 0.392*(yuv[:,1]-128)
    rgb[:,2] = 1.164*(yuv[:,0]-16) + 2.017*(yuv[:,1]-128)
    rgb = rgb/255
    
    return rgb




def load_sparse_tensor(filedir, device='cuda', order='rgb', scale=False):
    if filedir.endswith('ply'): coords, feats = read_ply_ascii(filedir, order=order) 
    if scale:
        coords = torch.tensor(coords).int() * scale
    else:
        coords = torch.tensor(coords).int()
    feats = torch.tensor(feats).float()/255.

    # feats = torch.round(feats)
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)

    return x


def array2vector_torch(array, step):
    array = array.long().cpu()
    step = array.max()+1
    vector = sum([array[:,i]*(step**i) for i in range(array.shape[-1])])

    return vector


def array2vector(array, step=None):
    array, step = array.long().clone(), step.long().clone()
    if array.min() < 0:
        min_value = array.min()
        array = array - min_value
        step = step - min_value

    assert array.min() >= 0 and array.max() - array.min() < step
    array, step = array.long(), step.long()
    vector = sum([array[:, i] * (step ** i) for i in range(array.shape[-1])])

    return vector

def isin(data, ground_truth):
    device = data.device
    if len(ground_truth) == 0:
        return torch.zeros([len(data)]).bool().to(device)
    data, ground_truth = data.cpu(), ground_truth.cpu()
    step = torch.max(data.max(), ground_truth.max()) + 1
    data = array2vector(data, step)
    ground_truth = array2vector(ground_truth, step)
    mask = np.isin(data.cpu().numpy(), ground_truth.cpu().numpy())

    return torch.Tensor(mask).bool().to(device)

def get_target_by_sp_tensor(out, coords_T):
    with torch.no_grad():
        def ravel_multi_index(coords, step):
            coords = coords.long()
            step = step.long()
            coords_sum = coords[:, 0] \
                         + coords[:, 1] * step \
                         + coords[:, 2] * step * step \
                         + coords[:, 3] * step * step * step
            return coords_sum

        step = max(out.C.cpu().max(), coords_T.max()) + 1
        out_sp_tensor_coords_1d = ravel_multi_index(out.C.cpu(), step)
        target_coords_1d = ravel_multi_index(coords_T, step)
        # test whether each element of a 1-D array is also present in a second array.
        target = np.in1d(out_sp_tensor_coords_1d, target_coords_1d)

        return torch.Tensor(target).bool()

def array2vector_sort(array):
    # 3D -> 1D by sum each dimension
    array = array.astype('int64')
    step = array.max() + 1
    vector = sum([array[:, i] * (step ** i) for i in range(array.shape[-1])])

    return vector

def sort_sparse_tensor(sparse_tensor):
    indices_sort = np.argsort(array2vector_sort(sparse_tensor.C.cpu().numpy()))
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort],
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0],
                                         device=sparse_tensor.device)

    return sparse_tensor_sort

def istopk(data, nums, rho=1.0):
    mask = torch.zeros(len(data), dtype=torch.bool)
    row_indices_per_batch = data._batchwise_row_indices
    for row_indices, N in zip(row_indices_per_batch, nums):
        k = int(min(len(row_indices), N*rho))
        _, indices = torch.topk(data.F[row_indices].squeeze().detach().cpu(), k)# must CPU.
        mask[row_indices[indices]]=True

    return mask.bool().to(data.device)


def knn_interpolation(x, new_coords, k):
    new_coords_C = new_coords.C[:,1:].unsqueeze(0).float()
    x_coords_C = x.C[:,1:].unsqueeze(0).float()
    x_attr = x.F.unsqueeze(0).float()
    x_nn = pytorch3d.ops.knn_points(new_coords_C, x_coords_C, K=k)
    knn_attribute = pytorch3d.ops.knn_gather(x_attr, x_nn.idx).squeeze(0)
    dists_sqrt = torch.sqrt(x_nn.dists)
    sigma = torch.mean(dists_sqrt, dim=2)
    sigma2 = sigma.unsqueeze(2) ** 2
    weight_ij = torch.exp(-(dists_sqrt / (2 * sigma2 + 1e-8))).squeeze(0)
    weight_ij_squeeze = weight_ij.unsqueeze(2)
    d_i = 1 / torch.sum(weight_ij, dim=1).unsqueeze(1)
    knn_attribute_weight = weight_ij_squeeze * knn_attribute

    interpolation_attribute = d_i * torch.sum(knn_attribute_weight, dim=1)

    return interpolation_attribute

def run_cmd(cmd):
    std_out_and_err = subprocess.Popen(cmd,
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE).communicate()
    return std_out_and_err
