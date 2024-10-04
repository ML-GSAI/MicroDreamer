from gs_renderer import GaussianModel, Renderer, MiniCam
from cam_utils import orbit_camera, OrbitCamera
from torchvision.utils import save_image
import numpy as np
import torch
import kiui
from sh_utils import SH2RGB
import os
from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
import torch.nn.functional as F
from copy import deepcopy
from glob import glob


#很小的高斯不剔除， 很大的高斯可能剔除， 如果它的投影面积大同时周围没有alpha，不剔除
#一个点影响的范围是否对alpha造成了贡献，如果没有则不剔除

def save_model(renderer, path):
    mesh = renderer.gaussians.extract_mesh(path, 0.2)

    # perform texture extraction
    print(f"[INFO] unwrap uv...")
    h = w = 512
    mesh.auto_uv()
    mesh.auto_normal()

    albedo = torch.zeros((h, w, 3), device="cuda", dtype=torch.float32)
    cnt = torch.zeros((h, w, 1), device="cuda", dtype=torch.float32)

    # self.prepare_train() # tmp fix for not loading 0123
    # vers = [0]
    # hors = [0]
    vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
    hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

    render_resolution = 512

    import nvdiffrast.torch as dr

    glctx = dr.RasterizeCudaContext()

    for ver, hor in zip(vers, hors):
        # render image
        pose = orbit_camera(ver, hor, 2)

        cur_cam = MiniCam(
            pose,
            render_resolution,
            render_resolution,
            np.deg2rad(49.1),
            np.deg2rad(49.1),
            0.01,
            100,
        )
        cam = OrbitCamera(512, 512, r=2, fovy=49.1)
        
        cur_out = renderer.render(cur_cam)

        rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

            
        # get coordinate in texture image
        pose = torch.from_numpy(pose.astype(np.float32)).to("cuda")
        proj = torch.from_numpy(cam.perspective.astype(np.float32)).to("cuda")

        v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T
        rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]

        alpha = (rast[0, ..., 3:] > 0).float()

        uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

        # use normal to produce a back-project mask
        normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
        mask = mask.view(-1)

        uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
        rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
        
        # update texture image
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
            h, w,
            uvs[..., [1, 0]] * 2 - 1,
            rgbs,
            min_resolution=256,
            return_count=True,
        )
        
        # albedo += cur_albedo
        # cnt += cur_cnt
        mask = cnt.squeeze(-1) < 0.1
        albedo[mask] += cur_albedo[mask]
        cnt[mask] += cur_cnt[mask]

    mask = cnt.squeeze(-1) > 0
    albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

    mask = mask.view(h, w)

    albedo = albedo.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    # dilate texture
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    inpaint_region = binary_dilation(mask, iterations=32)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        search_coords
    )
    _, indices = knn.kneighbors(inpaint_coords)

    albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

    mesh.albedo = torch.from_numpy(albedo).to("cuda")
    mesh.write(path)

    print(f"[INFO] save model to {path}.")

def render_save(renderer, pose, path=None):
    cam = MiniCam(pose, 512, 512, np.deg2rad(49), np.deg2rad(49), 0.01, 100)
    bg = torch.tensor([1,0,0],dtype=torch.float,device="cuda")
    out = renderer.render(cam,bg_color=bg)
    if path is not None:
        save_image(out["image"], path)
    return out["image"], out["alpha"]
    

# def alpha_save(path):
#     bg = torch.tensor([1,0,0],dtype=torch.float,device="cuda")
#     out = renderer.render(cam,bg_color=bg)
#     save_image(out["alpha"], path)

def drawPoint(img, point_img, path=None):
    color = torch.zeros((3, point_img.shape[0]), device = img.device)
    color[1, :] = 1
    img[:, point_img[:,1], point_img[:,0]] = color
    if path is not None:
        save_image(img, path)
    return img


def filter_out_once(renderer, pose):
    cam = MiniCam(pose, 512, 512, np.deg2rad(49), np.deg2rad(49), 0.01, 100)

    renderer_backup = deepcopy(renderer)

    scale = renderer_backup.gaussians.get_scaling
    xyz = renderer_backup.gaussians.get_xyz
    color = SH2RGB(renderer_backup.gaussians.get_features)
    opa = renderer_backup.gaussians.get_opacity

    prune_mask2 = torch.any(scale > 0.01, dim=1)

    prune_mask = torch.all(color > 1.0, dim=2)[:,0]
    prune_mask = torch.logical_and(prune_mask, prune_mask2)


    prune_xyz = xyz[prune_mask]
    homo = torch.ones((prune_xyz.shape[0], 1), device=prune_xyz.device)
    prune_xyz = torch.cat((prune_xyz, homo), dim=1)

    proj_mat = cam.full_proj_transform
    p_proj =  prune_xyz @ proj_mat
    point_img = p_proj[:,:2] / p_proj[:,[2]]
    point_img  = ((point_img + 1.0) * 512 / 2 - 0.5).round().int()

    img, alpha_before = render_save(renderer_backup, pose)
    #drawPoint(img, point_img, "1.png")
    prune_mask_backup = prune_mask.clone()
    renderer_backup.gaussians.prune_points_test(prune_mask_backup)

    _, alpha_after = render_save(renderer_backup, pose)

    alpha_delta = alpha_before - alpha_after
    alpha_delta = torch.where(alpha_delta > 0.1, 1.0, 0.0)
    
    torch.clamp_(point_img[:, 1], 0, alpha_delta.shape[1] - 1)
    torch.clamp_(point_img[:, 0], 0, alpha_delta.shape[2] - 1)
    
    prune_mask_contributed = torch.where(alpha_delta[0, point_img[:, 1], point_img[:, 0]] == 1.0, True, False)
    prune_mask_ind = torch.arange(prune_mask.shape[0], device=xyz.device)[prune_mask]
    prune_mask_ind = prune_mask_ind[prune_mask_contributed]
    prune_mask[:] = False
    prune_mask[prune_mask_ind] = True

    renderer.gaussians.prune_points_test(prune_mask)
    
    point_img = point_img[prune_mask_contributed]
    pointed_img = drawPoint(img, point_img)
    print(f"{point_img.shape[0]} guassians have been cleaned")
    return renderer, pointed_img

def filter_out(renderer):
    cams = []
    cams.append(orbit_camera(0, 0, 2))
    cams.append(orbit_camera(0, 180, 2))
    cams.append(orbit_camera(0, 90, 2))
    cams.append(orbit_camera(20, 180, 2))
    cams.append(orbit_camera(20, 90, 2))
    cams.append(orbit_camera(0, -90, 2))
    for ind, cam in enumerate(cams):
        _, pointed_img = filter_out_once(renderer, cam)

        

if __name__ == "__main__":

    ply_files = sorted(glob("./logs_v3d/*.ply"))

    renderer = Renderer(sh_degree=0)
    renderer.initialize(input=ply_files[13])

    filter_out(renderer)
