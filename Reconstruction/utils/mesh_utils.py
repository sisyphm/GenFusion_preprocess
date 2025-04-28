#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os, cv2
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
# import imageio # <<--- 주석 처리 또는 삭제
from easy_video import EasyWriter # <<--- 원래 임포트로 복구
# import easy_video.video as ev_video # <<--- 이전 시도 주석 처리
from utils.camera_utils import Camera
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image # <<--- 추가


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print(
        "post processing the mesh to have {} clusterscluster_to_kep".format(
            cluster_to_keep
        )
    )
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = (
            torch.tensor(
                [[W / 2, 0, 0, (W - 1) / 2], [0, H / 2, 0, (H - 1) / 2], [0, 0, 0, 1]]
            )
            .float()
            .cuda()
            .T
        )
        intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3, :3].T
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx=intrins[0, 2].item(),
            cy=intrins[1, 2].item(),
            fx=intrins[0, 0].item(),
            fy=intrins[1, 1].item(),
        )

        extrinsic = np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.names = []
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        self.image_gts = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(
            enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"
        ):
            if not isinstance(viewpoint_cam, Camera):
                cam = Camera(viewpoint_cam["cam_info"])
                gt = viewpoint_cam["image"][0].permute(2, 0, 1)
                self.image_gts.append(gt)
                self.names.append(viewpoint_cam["cam_info"].image_name[0])
            else:
                cam = viewpoint_cam
                self.names.append(f"{i:05d}.png")

            render_pkg = self.render(cam, self.gaussians)
            rgb = render_pkg["render"]
            alpha = render_pkg["rend_alpha"]
            normal = torch.nn.functional.normalize(render_pkg["rend_normal"], dim=0)
            depth = render_pkg["surf_depth"]
            depth_normal = render_pkg["surf_normal"]

            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn

        torch.cuda.empty_cache()
        c2ws = np.array(
            [
                np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy()))
                for cam in self.viewpoint_stack
            ]
        )
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        center = focus_point_fn(poses)
        self.radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(
        self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True
    ):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f"voxel_size: {voxel_size}")
        print(f"sdf_trunc: {sdf_trunc}")
        print(f"depth_truc: {depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, cam_o3d in tqdm(
            enumerate(to_cam_open3d(self.viewpoint_stack)),
            desc="TSDF integration progress",
        ):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(
                        np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255,
                        order="C",
                        dtype=np.uint8,
                    )
                ),
                o3d.geometry.Image(
                    np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")
                ),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
            )

            volume.integrate(
                rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic
            )

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
        return o3d.mesh
        """

        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
            compute per frame sdf
            """
            new_points = (
                torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
                @ viewpoint_cam.full_proj_transform
            )
            z = new_points[..., -1:]
            pix_coords = new_points[..., :2] / new_points[..., -1:]
            mask_proj = ((pix_coords > -1.0) & (pix_coords < 1.0) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(
                depthmap.cuda()[None],
                pix_coords[None, None],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).reshape(-1, 1)
            sampled_rgb = (
                torch.nn.functional.grid_sample(
                    rgbmap.cuda()[None],
                    pix_coords[None, None],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .reshape(3, -1)
                .T
            )
            sdf = sampled_depth - z
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(
            samples, inv_contraction, voxel_size, return_rgb=False
        ):
            """
            Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1 / (
                    2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9)
                )
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:, 0])
            for i, viewpoint_cam in tqdm(
                enumerate(self.viewpoint_stack), desc="TSDF integration progress"
            ):
                sdf, rgb, mask_proj = compute_sdf_perframe(
                    i,
                    samples,
                    depthmap=self.depthmaps[i],
                    rgbmap=self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[
                    :, None
                ]
                # update weight
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = self.radius * 2 / N
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction

        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R + 0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )

        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(
            torch.tensor(np.asarray(mesh.vertices)).float().cuda(),
            inv_contraction=None,
            voxel_size=voxel_size,
            return_rgb=True,
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        print("Exporting images for {} views...".format(len(self.rgbmaps)))
        os.makedirs(os.path.join(path, "gt"), exist_ok=True)
        os.makedirs(os.path.join(path, "renders"), exist_ok=True)
        os.makedirs(os.path.join(path, "vis"), exist_ok=True)
        cmap = matplotlib.colormaps.get_cmap("jet")
        for i in tqdm(range(len(self.rgbmaps)), desc="export images progress"):
            name = self.names[i]
            if len(self.image_gts) > 0:
                save_img_u8(
                    self.image_gts[i].cpu().permute(1, 2, 0).numpy(),
                    os.path.join(path, "gt", name),
                )

            save_img_u8(
                self.rgbmaps[i].cpu().permute(1, 2, 0).numpy(),
                os.path.join(path, "renders", name),
            )
            depth_map = self.depthmaps[i][0].cpu().numpy()
            save_img_f32(depth_map, os.path.join(path, "vis", name.replace(".png", ".tiff")))
            depth_map = (
                cmap(depth_map / (np.percentile(depth_map, 95) + 1e-5))
                * (depth_map > 1e-5)[..., None]
            )
            save_img_u8(depth_map, os.path.join(path, "vis", name))

    @torch.no_grad()
    def export_video_easy_test(self, gt_video_path, render_video_path, depth_video_path, fps=15):
        """EasyWriter를 사용하여 Test 세트의 GT, Rendered RGB, Depth를 비디오로 저장합니다."""
        print(f"Exporting videos using EasyWriter...")
        print(f"  - GT Video: {gt_video_path}")
        print(f"  - Rendered Video: {render_video_path}")
        print(f"  - Depth Video: {depth_video_path}")

        if not self.viewpoint_stack: # reconstruction이 호출되지 않은 경우
            print("  Warning: No reconstruction data found. Run reconstruction first.")
            return

        try:
            # 1. GT 비디오 저장 (self.image_gts 사용)
            if self.image_gts:
                print(f"  Processing {len(self.image_gts)} GT frames...")
                # (C, H, W) -> (H, W, C) NumPy 배열로 변환
                gt_frames = [img.permute(1, 2, 0).cpu().numpy() for img in self.image_gts]
                # [0, 1] float -> [0, 255] uint8 변환
                gt_frames_uint8 = [
                    np.clip(frame * 255, 0, 255).astype(np.uint8)
                    for frame in gt_frames
                ]
                gt_video_array = np.stack(gt_frames_uint8, axis=0) # (T, H, W, C)
                os.makedirs(os.path.dirname(gt_video_path), exist_ok=True)
                EasyWriter.writefile(filename=gt_video_path, video_array=gt_video_array, video_fps=fps, silent=False)
                print(f"    GT video saved successfully.")
            else:
                print("  Warning: No Ground Truth images found to create GT video.")

            # 2. Rendered RGB 비디오 저장 (self.rgbmaps 사용)
            if self.rgbmaps:
                print(f"  Processing {len(self.rgbmaps)} Rendered RGB frames...")
                # (C, H, W) -> (H, W, C) NumPy 배열로 변환
                render_frames = [img.permute(1, 2, 0).cpu().numpy() for img in self.rgbmaps]
                 # [0, 1] float -> [0, 255] uint8 변환
                render_frames_uint8 = [
                    np.clip(frame * 255, 0, 255).astype(np.uint8)
                    for frame in render_frames
                ]
                render_video_array = np.stack(render_frames_uint8, axis=0) # (T, H, W, C)
                os.makedirs(os.path.dirname(render_video_path), exist_ok=True)
                EasyWriter.writefile(filename=render_video_path, video_array=render_video_array, video_fps=fps, silent=False)
                print(f"    Rendered RGB video saved successfully.")
            else:
                print("  Warning: No Rendered RGB maps found to create video.")

            # 3. Rendered Depth 비디오 저장 (self.depthmaps 사용 - Grayscale)
            if self.depthmaps:
                print(f"  Processing {len(self.depthmaps)} Rendered Depth frames (Grayscale)... ")
                # 전체 깊이 값 범위 계산 (정규화용)
                all_depths = torch.stack(self.depthmaps).numpy() # (T, 1, H, W)
                valid_depths = all_depths[all_depths > 1e-5] # 유효한 깊이 값만
                if len(valid_depths) > 0:
                    min_depth, max_depth = np.min(valid_depths), np.percentile(valid_depths, 99) # 이상치 제외
                    if max_depth <= min_depth: max_depth = min_depth + 1e-5 # 분모 0 방지
                else:
                     min_depth, max_depth = 0, 1 # 유효 깊이 없으면 기본 범위

                # Convert normalized depth to colormap image
                cmap = matplotlib.colormaps.get_cmap('inferno_r') # Use reversed inferno colormap
                depth_video_frames = []
                for i in range(all_depths.shape[0]):
                    depth_map = all_depths[i, 0] # (H, W)
                    # 정규화 [0, 1]
                    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
                    normalized_depth = np.clip(normalized_depth, 0, 1)
                    # Apply colormap (returns RGBA float 0-1), take RGB, scale to 0-255 uint8
                    colored_depth = (cmap(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
                    depth_video_frames.append(colored_depth)
                depth_video_array = np.stack(depth_video_frames, axis=0) # (T, H, W, C)

                # Write video file
                print(f"    Saving depth video to: {depth_video_path}")
                os.makedirs(os.path.dirname(depth_video_path), exist_ok=True)
                EasyWriter.writefile(filename=depth_video_path, video_array=depth_video_array, video_fps=fps, silent=False)
                print(f"    Rendered Depth video (Grayscale) saved successfully.")
            else:
                print("  Warning: No Rendered Depth maps found to create video.")

        except ImportError:
             print(f"  Error: Could not import EasyWriter from easy_video. Please ensure 'easy_video' library is installed correctly.")
        except Exception as e:
            print(f"  Error during EasyWriter video export: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def export_video(self, path):
        disparities = 1.0 / (1 + torch.cat(self.depthmaps).cpu().numpy()) * 255
        disparities = [
            cv2.applyColorMap(item.astype("uint8"), cv2.COLORMAP_JET)
            for item in disparities
        ]
        renderings = [
            item.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
            for item in self.rgbmaps
        ]
        renderings = [item.astype("uint8") for item in renderings]

        imageio.mimwrite(f"{path}_depth.mp4", disparities, fps=30, quality=10)
        imageio.mimwrite(f"{path}_rgb.mp4", renderings, fps=30, quality=10)
