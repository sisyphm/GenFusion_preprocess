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

import sys
import torch
from datasets import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import DataParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import to_cam_open3d, post_process_mesh
from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    dp = DataParams(parser)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--voxel_size", default=-1.0, type=float, help="Mesh: voxel size for TSDF"
    )
    parser.add_argument(
        "--depth_trunc", default=-1.0, type=float, help="Mesh: Max depth range for TSDF"
    )
    parser.add_argument(
        "--sdf_trunc", default=-1.0, type=float, help="Mesh: truncation value for TSDF"
    )
    parser.add_argument(
        "--num_cluster",
        default=50,
        type=int,
        help="Mesh: number of connected clusters to export",
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="Mesh: using unbounded mode for meshing",
    )
    parser.add_argument(
        "--mesh_res",
        default=1024,
        type=int,
        help="Mesh: resolution for unbounded mesh extraction",
    )
    parser.add_argument("--mono_depth", action="store_true")
    parser.add_argument(
        "--downsample_factor",
        default=1,
        type=int,
        help="Downsample factor for the input images",
    )
    parser.add_argument("--video_only", action="store_true", help="only render video")
    parser.add_argument("--camera_path_file", type=str, default=None)
    parser.add_argument("--output_video_base_dir", type=str, default="./render_videos", help="Base directory to save the output videos.")
    parser.add_argument("--video_fps", type=int, default=15, help="FPS for the generated videos.")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    scene_name = os.path.basename(os.path.normpath(args.model_path))
    print(f"Detected scene name: {scene_name}")

    dataset_args = dp.extract(args)
    args.dataset = dataset_args
    model_args = model.extract(args)
    iteration = args.iteration
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(model_args.sh_degree)
    scene = Scene(
        args.model_path, args, gaussians, load_iteration=iteration, shuffle=False
    )
    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    if not args.skip_mesh:
        print("Mesh extraction logic (if enabled) remains unchanged.")

    print("\nProcessing Test set for video export...")
    if len(scene.valset) > 0:
        print(f"Found {len(scene.valset)} test views. Performing reconstruction...")
        gaussExtractor.reconstruction(
            [scene.getTestInstant() for _ in range(len(scene.valset))]
        )

        gt_video_dir = os.path.join(args.output_video_base_dir, "gt_videos")
        render_video_dir = os.path.join(args.output_video_base_dir, "rendered_videos")
        depth_video_dir = os.path.join(args.output_video_base_dir, "rendered_depth_videos")

        gt_video_path = os.path.join(gt_video_dir, f"{scene_name}.mp4")
        render_video_path = os.path.join(render_video_dir, f"{scene_name}.mp4")
        depth_video_path = os.path.join(depth_video_dir, f"{scene_name}.mp4")

        gaussExtractor.export_video_easy_test(
            gt_video_path=gt_video_path,
            render_video_path=render_video_path,
            depth_video_path=depth_video_path,
            fps=args.video_fps
        )
    else:
        print("No test views found in the scene. Skipping video export.")

    print(f"\nFinished processing scene: {scene_name}")
