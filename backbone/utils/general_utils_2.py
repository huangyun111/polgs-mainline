#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
from pytorch3d.ops import knn_points
import pymeshlab
import open3d as o3d
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.io import IO
from plyfile import PlyData, PlyElement
from tqdm import tqdm

def cutoff_act(x, low=0.1, high=12):
    return low + (high - low) * torch.sigmoid(x)

def cutoff_act_inverse(x, low=0.1, high=12):
    x_ = (x - low) / (high - low)
    return torch.log(x_ / (1 - x_))

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def normal2rotation(n):
    # construct a random rotation matrix from normal
    # it would better be positive definite and orthogonal
    n = torch.nn.functional.normalize(n)
    # w0 = torch.rand_like(n)
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape).to(n.device)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.cross(n, R0)
    
    # i = 7859
    # print(R1[i])
    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    # print(R1[i])
    R = torch.stack([R0, R1, n], -1)
    # print(R[i], torch.det(R).sum(), torch.trace(R[i]))
    q = rotmat2quaternion(R)
    # print(q[i], torch.norm(q[i]))
    # R = quaternion2rotmat(q)
    # print(R[i])
    # for i in range(len(q)):
    #     if torch.isnan(q[i].sum()):
    #         print(i)
    # exit()
    return q

def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    # R = torch.eye(4).expand([len(q), 4, 4]).to(q.device)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
    ], -1).reshape([len(q), 3, 3]);
    return R

def rotmat2quaternion(R, normalize=False):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    # print(torch.sum(torch.isnan(r)))
    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)

    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q


def knn_pcl(pcl0, pcl1, feat, K):
    nn_dist, nn_idx, nn_vtx = knn_points(pcl0[..., :3][None], pcl1[..., :3][None], K=K+1, return_nn=True)
    nn_dist = nn_dist[0, :, 1:]
    nn_idx = nn_idx[0, :, 1:]
    nn_vtx = nn_vtx[0, :, 1:]
    nn_vtx = torch.mean(nn_vtx, axis=1)
    nn_feat = torch.mean(feat[nn_idx], axis=1)
    return nn_vtx, nn_feat



def poisson_mesh(path, vtx, normal, color, depth, thrsh):
    # GPU-first Poisson meshing using Open3D tensor API; falls back to CPU for Poisson solve
    pbar = tqdm(total=4)
    pbar.update(1)
    pbar.set_description('Poisson meshing')

    torch_device = vtx.device
    o3d_device = o3d.core.Device('CUDA:0') if torch_device.type == 'cuda' and o3d.core.cuda.is_available() else o3d.core.Device('CPU:0')

    # build Open3D point cloud on GPU when available
    pcd = o3d.t.geometry.PointCloud(o3d_device)
    pcd.point['positions'] = o3d.core.Tensor(vtx.detach().cpu().numpy(), device=o3d_device)
    pcd.point['normals'] = o3d.core.Tensor(normal.detach().cpu().numpy(), device=o3d_device)
    pcd.point['colors'] = o3d.core.Tensor(color.detach().cpu().numpy().clip(0, 1), device=o3d_device)

    # Poisson reconstruction (currently CPU in Open3D)
    mesh_legacy, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd.to_legacy(), depth=depth)
    mesh_plain = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy, device=o3d_device)
    o3d.t.io.write_triangle_mesh(path + '_plain.ply', mesh_plain)

    pbar.update(1)
    pbar.set_description('Mesh refining')
    vert = torch.utils.dlpack.from_dlpack(mesh_plain.vertex['positions'].to_dlpack()).to(torch_device)
    nn_dist, nn_idx, _ = knn_points(vert[None], vtx[None], K=4)
    nn_dist = nn_dist[0]
    nn_idx = nn_idx[0]
    nn_color = torch.mean(color[nn_idx], dim=1)

    # set vertex colors and per-vertex quality (distance to closest sampled point)
    mesh_plain.vertex['colors'] = o3d.core.Tensor(nn_color.clip(0, 1).detach().cpu().numpy(), device=o3d_device)
    mesh_plain.vertex['quality'] = o3d.core.Tensor(nn_dist[:, 0].detach().cpu().numpy(), device=o3d_device)

    pbar.update(1)
    pbar.set_description('Mesh cleaning')
    # prune vertices with large reconstruction error
    mask = nn_dist[:, 0] > thrsh
    mesh_clean_legacy = mesh_plain.to_legacy()
    # mesh_clean_legacy.remove_vertices_by_mask(mask.detach().cpu().numpy())
    # mesh_clean_legacy.remove_unreferenced_vertices()
    # mesh_clean_legacy.remove_degenerate_triangles()
    # mesh_clean_legacy.remove_duplicated_triangles()
    # mesh_clean_legacy.remove_duplicated_vertices()

    # mild smoothing to reduce pruning artifacts (use legacy API for smoothing)
    mesh_clean_legacy.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)
    mesh_clean_legacy.compute_vertex_normals()
    
    # convert back to tensor mesh for GPU operations if needed
    mesh_clean = o3d.t.geometry.TriangleMesh.from_legacy(mesh_clean_legacy, device=o3d_device)
    o3d.t.io.write_triangle_mesh(path + '_pruned.ply', mesh_clean)

    pbar.update(1)
    pbar.close()

env_rayd1 = None
def init_envrayd1(H,W):
    i, j = np.meshgrid(
        np.linspace(-np.pi, np.pi, W, dtype=np.float32),
        np.linspace(0, np.pi, H, dtype=np.float32),
        indexing='xy'
    )
    xy1 = np.stack([i, j], axis=2)
    z = np.cos(xy1[..., 1])
    x = np.sin(xy1[..., 1])*np.cos(xy1[...,0])
    y = np.sin(xy1[..., 1])*np.sin(xy1[...,0])
    global env_rayd1
    env_rayd1 = torch.tensor(np.stack([x,y,z], axis=-1)).cuda()

def get_env_rayd1(H,W):
    if env_rayd1 is None:
        init_envrayd1(H,W)
    return env_rayd1


env_rayd2 = None
def init_envrayd2(H,W):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            # indexing='ij')
                            )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    global env_rayd2
    env_rayd2 = reflvec

def get_env_rayd2(H,W):
    if env_rayd2 is None:
        init_envrayd2(H,W)
    return env_rayd2