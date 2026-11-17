import numpy as np
import time
import torch
import torch.nn.functional as F
def mvas_net(depth,ray_d,ray_o,object_mask,device,dataset,depth_list, threshold, training):
    if training:
        # view_directions, camera_centers, object_mask, idx = ray_d[:128,...], ray_o[:128,...], mask[:128,...], idx[:128,...]
        num_pixels, _ = ray_d.shape

        points = ray_o[None,...].repeat(num_pixels, 1) + depth * ray_d  # (num_pixels, 3)
        surface_mask = object_mask.squeeze(-1) > 0.5  # (num_pixels,)
        
        with torch.no_grad():

            points_homo = torch.cat(
                (points[surface_mask], torch.ones((surface_mask.sum(), 1), dtype=float, device=device)), -1).float()
            # project points onto all image planes
            # (num_cams, 3, 4) x (4, num_points)->  (num_cams, 3, num_points)
            pixel_coordinates_homo = torch.einsum("ijk, kp->ijp", dataset['projection_matrices'],
                                                    points_homo.T).cpu().detach().numpy()
            pixel_coordinates_xx = (pixel_coordinates_homo[:, 0, :] / (
                        pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)
            pixel_coordinates_yy = (pixel_coordinates_homo[:, 1, :] / (
                        pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)

            # opencv convention to numpy axis convention
            #  (top left) ----> x    =>  (top left) ---> axis 1
            #    |                           |
            #    |                           |
            #    |                           |
            #    y                         axis 0
            index_axis0 = np.round(pixel_coordinates_yy)  # (num_points, num_cams)
            index_axis1 = np.round(pixel_coordinates_xx)  # (num_points, num_cams)
            # index_axis0 = np.round(dataset.img_height-pixel_coordinates_yy)  # (num_points, num_cams)
            # index_axis1 = np.round(dataset.img_width-pixel_coordinates_xx)  # (num_points, num_cams)
            index_axis0 = np.clip(index_axis0, int(0), int(dataset['H'] - 1)).astype(
                np.int64)  # (num_points, num_cams)
            index_axis1 = np.clip(index_axis1, int(0), int(dataset['W'] - 1)).astype(np.int64)

            # (num_cams, img_height, img_width, 3) -> (num_points, num_cams, 3)
            num_cams = index_axis0.shape[1]
            tangent_vectors_all_view_list = []
            tangent_vectors_half_pi_all_view_list = []
            depth_values_all_view_list = []
            for cam_idx in range(num_cams):
                azimuth_angles = dataset['azimuths'][cam_idx,
                                                                index_axis0[:, cam_idx],
                                                                index_axis1[:, cam_idx]]  # (num_surface_points)
                
                depth_values = depth_list[cam_idx,
                                                            index_axis0[:, cam_idx],
                                                            index_axis1[:, cam_idx]] # (num_surface_points)

                R_list = dataset['poses'][cam_idx]
                r1 = R_list[0, :3]
                r2 = R_list[1, :3]
                tangent_vectors_all_view_list.append(
                    r1 * torch.sin(azimuth_angles[:, None]) - r2 * torch.cos(azimuth_angles[:, None]))
                tangent_vectors_half_pi_all_view_list.append(r1 * torch.sin(azimuth_angles[:, None] + np.pi / 2) -
                                                                r2 * torch.cos(azimuth_angles[:, None] + np.pi / 2))
                depth_values_all_view_list.append(depth_values)

            tangent_vectors_all_view = torch.stack(tangent_vectors_all_view_list, dim=1).to(
                device)  # (num_points, num_cams, 3)
            tangent_vectors_half_pi_all_view = torch.stack(tangent_vectors_half_pi_all_view_list, dim=1).to(
                device)  # (num_points, num_cams, 3)
            depth_values_all_view = torch.stack(depth_values_all_view_list, dim=1).to(device)
        visibility_mask = visibility_from_depth(points[surface_mask], dataset['camera_centers'], depth_values_all_view, threshold=threshold)

        output = {
            # the mask indicating which pixels hit the zero level set (for mask loss)
            "tangent_vectors_all_view": tangent_vectors_all_view,
            "tangent_vectors_all_view_half_pi": tangent_vectors_half_pi_all_view,
            "visibility_mask": visibility_mask,
            "surface_mask": surface_mask,
        }

    else:
        output = {
            "tangent_vectors_all_view": 0,
            "tangent_vectors_all_view_half_pi": 0,
            "visibility_mask": 0,
            "surface_mask": 0,
            "sdf_output": 0,
            'grad_theta': 0,
            'surface_normal': 0
        }

    return output


def visibility_from_depth(points, camera_centers, depth_list, threshold=0.01):
    """
    points: (n_points, 3)
    camera_centers: (n_cameras, 3)
    depth_values: (n_points, n_cameras)
    """
    n_points, _ = points.shape
    n_cameras, _ = camera_centers.shape

    points_expanded = points[:, None, :].expand(n_points, n_cameras, 3)
    camera_centers_expanded = camera_centers[None, :, :].expand(n_points, n_cameras, 3)
    depth = torch.linalg.norm(points_expanded - camera_centers_expanded, dim=-1)
    

    visibility_mask = (-threshold < depth_list.reshape(n_points, n_cameras) - depth)
    # breakpoint()
    return visibility_mask.detach()

def get_azimuth_loss(surface_normal, outputs):
    TSC_loss =get_half_pi_tangent_space_consistency_loss(surface_normal,
                                                                    outputs["tangent_vectors_all_view"],
                                                                    outputs[
                                                                        "tangent_vectors_all_view_half_pi"],
                                                                    outputs["visibility_mask"],
                                                                    outputs["surface_mask"] 
                                                                    )
    

    return TSC_loss
def get_half_pi_tangent_space_consistency_loss(normals, tangents_all_view, tangents_all_view_pi2,
                                                   visibility_mask, surface_mask):
    normals = F.normalize(normals,dim=-1) # (65536,3)
    normals = normals[surface_mask]  # (num_surface_points, 3)
    not_nan_mask = ~torch.isnan(tangents_all_view.sum(-1))

    visibility_mask = visibility_mask & not_nan_mask  # (43345,6)
    tangents_all_view[torch.isnan(tangents_all_view)] = 1  # (num_surface_points, num_views, 3)
    tangents_all_view_pi2[torch.isnan(tangents_all_view_pi2)] = 1
    num_visible_views = visibility_mask.sum(-1)

    # compute the tangent space consistency loss for each surface point in each view, considering both possibilities
    
    loss_1= ((normals.reshape(-1,1,3) * tangents_all_view).sum(-1)) ** 2  
    # loss_1 = loss_1 * weights

    loss_2 = ((normals.reshape(-1,1,3) * tangents_all_view_pi2).sum(-1)) ** 2
    # loss_2 = loss_2 * weights

    loss_1 = loss_1 * visibility_mask
    loss_2 = loss_2 * visibility_mask
    visible_view_mask = num_visible_views > 0
    # sum over all views for each surface point
    loss = loss_1 * loss_2 # (num_surface_points, num_views)

    loss = loss[visible_view_mask].sum(-1) / num_visible_views[visible_view_mask]
    # sum over all surface points
    loss = loss.sum() / float(visible_view_mask.shape[0])
    return loss

def normalize(v):
    import torch
    return torch.nn.functional.normalize(v, dim=-1)

def normalize_np(v):
    return  v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),
                     1e-12)

def stokes_fac_from_normal(rays_o, rays_d, normal, 
                           train_mode=False,
                           ret_spec=False,
                           clip_spec=False):
    import torch

    # Add singleton dimension for Num_lights
    rays_d = rays_d[..., None, :]
    normal = normal[..., None, :]

    cos = torch.cos 
    sin = torch.sin 
    acos = torch.acos
    atan2 = torch.atan2
    sqrt = torch.sqrt
    normize = normalize
    stack = torch.stack
    acos = lambda x: torch.acos(torch.clamp(x, min=-1.+1e-7, max=1.-1e-7))
    dot = lambda x, y: (x * y).sum(-1, keepdim=True)
    clamp = lambda x, y: torch.clamp(x, min=y)
    mask_fn = lambda x, mask: torch.where(mask, x, torch.zeros_like(x))
    cross = lambda x, y: torch.cross(x, y.broadcast_to(x.shape), dim=-1)

    # Helper variables    
    eta = 1.5
    n = normal
    o = -rays_d
    # h = n 
    dotno = dot(n, o)
    n_o = normize(n - dotno * o)
    # h_o = n_o

    x_o = normize(stack([-o[..., 1], o[..., 0], torch.zeros_like(o[..., 2])], -1))
    y_o = cross(x_o, o)
    phi_o = atan2(-dot(n_o, x_o), dot(n_o, y_o))
    psi_o = phi_o

    eta_i_1, eta_i_2 = 1.0, eta
    theta_i_1 = acos(dotno)
    theta_i_2 = acos(sqrt(clamp(1 - (sin(theta_i_1) / eta) ** 2, 1e-7)))

    eta_o_1, eta_o_2 = eta, 1.0
    theta_o_2 = acos(dotno)
    theta_o_1 = acos(sqrt(clamp(1 - (sin(theta_o_2) / eta) ** 2, 1e-7)))

    theta_d = acos(dotno)
    eta_r_1, eta_r_2 = 1.0, eta
    theta_r_1 = theta_d
    theta_r_2 = acos(sqrt(clamp(1 - (sin(theta_r_1) / eta) ** 2, 1e-7)))

    T_i__perp = (2 * eta_i_1 * cos(theta_i_1)) ** 2 / clamp((eta_i_1 * cos(theta_i_1) + eta_i_2 * cos(theta_i_2)) ** 2, 1e-7)
    T_i__perp *= (cos(theta_i_1) > 1e-7)
    T_i__para = (2 * eta_i_1 * cos(theta_i_1)) ** 2 / clamp((eta_i_1 * cos(theta_i_2) + eta_i_2 * cos(theta_i_1)) ** 2, 1e-7)
    T_i__para *= (cos(theta_i_1) > 1e-7)

    T_o__perp = (2 * eta_o_1 * cos(theta_o_1)) ** 2 / clamp((eta_o_1 * cos(theta_o_1) + eta_o_2 * cos(theta_o_2)) ** 2, 1e-7)
    T_o__para = (2 * eta_o_1 * cos(theta_o_1)) ** 2 / clamp((eta_o_1 * cos(theta_o_2) + eta_o_2 * cos(theta_o_1)) ** 2, 1e-7)
    T_o__plus, T_o__min = 0.5 * (T_o__perp + T_o__para), 0.5 * (T_o__perp - T_o__para)

    R__perp = (eta_r_1 * cos(theta_r_1) - eta_r_2 * cos(theta_r_2)) ** 2 / clamp((eta_r_1 * cos(theta_r_1) + eta_r_2 * cos(theta_r_2)) ** 2, 1e-7)
    R__para = (eta_r_1 * cos(theta_r_2) - eta_r_2 * cos(theta_r_1)) ** 2 / clamp((eta_r_1 * cos(theta_r_2) + eta_r_2 * cos(theta_r_1)) ** 2, 1e-7)
    R__plus, R__min = 0.5 * (R__perp + R__para), 0.5 * (R__perp - R__para)

    stokes_diff_fac = stack([torch.ones_like(T_o__min), T_o__min / T_o__plus * cos(2 * phi_o), -T_o__min / T_o__plus * sin(2 * phi_o)], -1)
    stokes_spec_fac = stack([torch.ones_like(R__plus), R__min / R__plus * cos(2 * psi_o), -R__min / R__plus * sin(2 * psi_o)], -1)

    if clip_spec:
        spec_mask = dotno > 1e-7
        stokes_spec_fac = mask_fn(stokes_spec_fac, spec_mask[..., None])
        R__plus = mask_fn(R__plus, spec_mask)

    return stokes_diff_fac, stokes_spec_fac, R__plus[..., None]