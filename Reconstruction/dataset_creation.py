import os
import subprocess
import argparse
import multiprocessing
import torch
from functools import partial
from tqdm import tqdm
import shutil
# from easy_video import EasyWriter # <<--- 제거
# import glob                     # <<--- 제거
# import shutil                   # <<--- 제거
# from PIL import Image           # <<--- 제거
# import numpy as np              # <<--- 제거


def worker_process_scene(args, base_data_dir, base_output_dir, scene_info):
    """Worker function to process a single scene on a specific GPU."""
    scene_name, gpu_id = scene_info

    # Get the absolute path of the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct absolute paths for train.py and render.py
    train_script_path = os.path.join(script_dir, "train.py")
    render_script_path = os.path.join(script_dir, "render.py")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # --- Restore original data path logic --- 
    # Assuming pre-prepared data follows the structure: <base_data_dir>/<scene_name>/colmap
    scene_data_path = os.path.join(base_data_dir, scene_name, "colmap")
    scene_output_path = os.path.join(base_output_dir, scene_name) # Intermediate output path

    print(f"[GPU {gpu_id}] --- Processing scene: {scene_name} ---")
    print(f"[GPU {gpu_id}] Data directory: {scene_data_path}") # Using corrected data path logic
    print(f"[GPU {gpu_id}] Output directory: {scene_output_path}")
    # print(f"[GPU {gpu_id}] Train script path: {train_script_path}") # Debug print
    # print(f"[GPU {gpu_id}] Render script path: {render_script_path}") # Debug print

    os.makedirs(scene_output_path, exist_ok=True)

    # Use the restored scene_data_path for train.py
    train_cmd = [
        "python", train_script_path, # Use absolute path
        "--data_dir", scene_data_path, # <<<--- RESTORED PATH HERE
        "-m", scene_output_path,
        "--iterations", str(args.iterations),
        "--test_iterations", "0",
        "--diffusion_ckpt", args.diffusion_ckpt,
        "--diffusion_config", args.diffusion_config,
        "--num_frames", str(args.num_frames),
        "--outpaint_type", args.outpaint_type,
        "--start_diffusion_iter", str(args.start_diffusion_iter),
        "--sparse_view", str(args.sparse_view),
        "--downsample_factor", str(args.downsample_factor),
        "--diffusion_resize_width", str(args.diffusion_resize_width),
        "--diffusion_resize_height", str(args.diffusion_resize_height),
        "--diffusion_crop_width", str(args.diffusion_crop_width),
        "--diffusion_crop_height", str(args.diffusion_crop_height),
        "--patch_size", str(args.patch_size[0]), str(args.patch_size[1]),
        "--opacity_reset_interval", str(args.opacity_reset_interval),
        "--lambda_dist", str(args.lambda_dist),
        "--lambda_reg", str(args.lambda_reg),
        "--lambda_dssim", str(args.lambda_dssim),
        "--densify_from_iter", str(args.densify_from_iter),
        "--unconditional_guidance_scale", str(args.unconditional_guidance_scale),
    ]
    if args.quiet:
        train_cmd.append("--quiet")
    if args.detect_anomaly:
        train_cmd.append("--detect_anomaly")

    render_cmd = [
        "python", render_script_path, # Use absolute path
        "-m", scene_output_path,
        "--output_video_base_dir", args.output_video_base_dir, # 비디오 저장 기본 경로 전달
        "--video_fps", str(args.video_fps) # 비디오 FPS 전달
    ]
    if args.quiet:
        render_cmd.append("--quiet")

    print(f"\n[GPU {gpu_id}] Running train command for {scene_name}:")
    try:
        subprocess.run(train_cmd, check=True, env=env, capture_output=args.quiet, text=True, stdout=subprocess.DEVNULL)
        print(f"\n[GPU {gpu_id}] Training finished for scene: {scene_name}")
    except subprocess.CalledProcessError as e:
        print(f"[GPU {gpu_id}] Error during training for scene {scene_name}:")
        print(f"Command: {' '.join(e.cmd)}")
        if e.stdout: print(f"stdout:\n{e.stdout}")
        if e.stderr: print(f"stderr:\n{e.stderr}")
        return

    print(f"\n[GPU {gpu_id}] Running render command for {scene_name} (Outputting videos directly):")

    try:
        subprocess.run(render_cmd, check=True, env=env, capture_output=args.quiet, text=True, stdout=subprocess.DEVNULL)
        print(f"\n[GPU {gpu_id}] Rendering (and video export) finished for scene: {scene_name}")

        print(f"[GPU {gpu_id}] Cleaning up intermediate output directory: {scene_output_path}")
        try:
            if os.path.isdir(scene_output_path):
                shutil.rmtree(scene_output_path)
                print(f"[GPU {gpu_id}] Successfully removed intermediate directory.")
            else:
                print(f"[GPU {gpu_id}] Intermediate directory not found, skipping cleanup: {scene_output_path}")
        except Exception as e:
            print(f"[GPU {gpu_id}] Error during cleanup of {scene_output_path}: {e}")

    except subprocess.CalledProcessError as e:
        print(f"[GPU {gpu_id}] Error during rendering or video export for scene {scene_name}:")
        print(f"Command: {' '.join(e.cmd)}")
        if e.stdout: print(f"stdout:\n{e.stdout}")
        if e.stderr: print(f"stderr:\n{e.stderr}")

    print(f"[GPU {gpu_id}] --- Finished processing scene: {scene_name} ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train and render (outputting videos directly) for multiple scenes using specified GPUs.")

    # parser.add_argument("--base_data_dir", type=str, default="/root/code/enhance_4dgs_with_vdm/GenFusion/Reconstruction/data/datasets--Inception3D--GenFusion_DL3DV_24Benchmark/snapshots/5bae7e90bb2613df3e05eba5e9060f54914eb29b", help="Base directory containing all scene folders.")
    parser.add_argument("--base_data_dir", type=str, default="/root/code/GenFusion_preprocess/test_data", help="Base directory containing all scene folders.")
    parser.add_argument("--base_output_dir", type=str, default="./data_output", help="Base directory where output for each scene will be stored.")
    parser.add_argument('--iterations', type=int, default=7000)
    parser.add_argument('--test_iterations', type=int, default=7000)
    parser.add_argument('--diffusion_ckpt', type=str, default="./diffusion_ckpt/epoch=59-step=34000.ckpt")
    parser.add_argument('--diffusion_config', type=str, default="./generation_infer.yaml")
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--outpaint_type', type=str, default='crop')
    parser.add_argument('--start_diffusion_iter', type=int, default=3000)
    parser.add_argument('--sparse_view', type=int, default=0)
    parser.add_argument('--downsample_factor', type=int, default=2)
    parser.add_argument('--diffusion_resize_width', type=int, default=960)
    parser.add_argument('--diffusion_resize_height', type=int, default=512)
    parser.add_argument('--diffusion_crop_width', type=int, default=960)
    parser.add_argument('--diffusion_crop_height', type=int, default=512)
    parser.add_argument('--patch_size', nargs=2, type=int, default=[480, 256])
    parser.add_argument('--opacity_reset_interval', type=int, default=9000)
    parser.add_argument('--lambda_dist', type=float, default=0.0)
    parser.add_argument('--lambda_reg', type=float, default=0.5)
    parser.add_argument('--lambda_dssim', type=float, default=0.8)
    parser.add_argument('--densify_from_iter', type=int, default=1000)
    parser.add_argument('--unconditional_guidance_scale', type=float, default=3.2)
    parser.add_argument("--quiet", action="store_true", help="Suppress command output.")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=None, help="List of GPU IDs to use (e.g., 0 1 3). Defaults to all available CUDA GPUs.")
    parser.add_argument("--video_fps", type=int, default=15, help="FPS for the generated videos.")
    parser.add_argument("--output_video_base_dir", type=str, default="./DL3DV-dataset", help="Base directory to save the output videos (e.g., ./DL3DV-dataset).")
    parser.add_argument("--start_idx", type=float, default=0.0, help="Starting fraction of scenes to process (e.g., 0.0 for the beginning).")
    parser.add_argument("--end_idx", type=float, default=1.0, help="Ending fraction of scenes to process (e.g., 0.1 for the first 10%%). Exclusive.")

    args = parser.parse_args()

    if not (0.0 <= args.start_idx <= 1.0 and 0.0 <= args.end_idx <= 1.0 and args.start_idx <= args.end_idx):
        print("Error: Invalid arguments. Ensure 0.0 <= start_idx <= end_idx <= 1.0")
        exit(1)

    try:
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            print("오류: 사용 가능한 CUDA GPU가 없습니다. PyTorch 설치 및 CUDA 설정을 확인하세요.")
            exit(1)
        all_gpu_ids = list(range(available_gpus))
    except Exception as e:
        print(f"CUDA 장치 확인 중 오류 발생: {e}")
        print("PyTorch가 CUDA 지원과 함께 올바르게 설치되었는지 확인하세요.")
        exit(1)

    if args.gpu_ids is None:
        gpu_ids_to_use = all_gpu_ids
        print(f"사용할 GPU ID가 지정되지 않았습니다. 사용 가능한 모든 GPU {gpu_ids_to_use}를 사용합니다.")
    else:
        invalid_ids = [gid for gid in args.gpu_ids if gid not in all_gpu_ids]
        if invalid_ids:
            print(f"오류: 잘못된 GPU ID가 포함되어 있습니다: {invalid_ids}. 사용 가능한 ID: {all_gpu_ids}")
            exit(1)
        gpu_ids_to_use = sorted(list(set(args.gpu_ids)))
        print(f"지정된 GPU ID {gpu_ids_to_use}를 사용합니다.")

    num_gpus_to_use = len(gpu_ids_to_use)
    if num_gpus_to_use == 0:
        print("오류: 사용할 GPU가 없습니다.")
        exit(1)

    if not os.path.isdir(args.base_data_dir):
        print(f"오류: 기본 데이터 디렉토리를 찾을 수 없습니다: {args.base_data_dir}")
        exit(1)

    os.makedirs(args.base_output_dir, exist_ok=True)

    # --- 추가: 비디오 저장 기본 디렉토리 생성 ---
    os.makedirs(os.path.join(args.output_video_base_dir, "gt_videos"), exist_ok=True)
    os.makedirs(os.path.join(args.output_video_base_dir, "rendered_videos"), exist_ok=True)
    os.makedirs(os.path.join(args.output_video_base_dir, "rendered_depth_videos"), exist_ok=True)

    all_scenes = []
    print("유효한 scene 검색 중...")
    for item_name in sorted(os.listdir(args.base_data_dir)):
        if item_name == '.cache' or item_name.startswith('.'):
            continue

        item_path = os.path.join(args.base_data_dir, item_name)

        if os.path.isdir(item_path):
            all_scenes.append(item_name)
        else:
            pass

    num_total_scenes = len(all_scenes)
    if num_total_scenes == 0:
        print("처리할 유효한 scene을 찾지 못했습니다.")
        exit(0)

    print(f"{num_total_scenes}개의 유효한 scene을 찾았습니다.")

    start_index = int(num_total_scenes * args.start_idx)
    end_index = int(num_total_scenes * args.end_idx)
    end_index = min(end_index, num_total_scenes)

    scenes_to_process = all_scenes[start_index:end_index]
    num_scenes_to_process = len(scenes_to_process)

    if num_scenes_to_process == 0:
        print(f"선택된 범위 [{args.start_idx:.2f}, {args.end_idx:.2f})에 해당하는 처리할 scene이 없습니다.")
        exit(0)

    print(f"총 {num_total_scenes}개 중, 범위 [{args.start_idx:.2f}, {args.end_idx:.2f})에 해당하는 {num_scenes_to_process}개의 scene을 처리합니다 (인덱스 {start_index}부터 {end_index-1}까지).")

    scene_gpu_assignments = []
    for i, scene_name in enumerate(scenes_to_process):
        assigned_gpu_id = gpu_ids_to_use[i % num_gpus_to_use]
        scene_gpu_assignments.append((scene_name, assigned_gpu_id))

    print(f"\n{num_gpus_to_use}개의 지정된 GPU({gpu_ids_to_use})에서 병렬 처리 시작...")

    process_func = partial(worker_process_scene, args, args.base_data_dir, args.base_output_dir)

    with multiprocessing.Pool(processes=num_gpus_to_use) as pool:
        print(f"Processing {len(scene_gpu_assignments)} scenes...")
        list(tqdm(pool.imap_unordered(process_func, scene_gpu_assignments), total=len(scene_gpu_assignments), desc="Overall Scene Progress"))

    print("\n--- 모든 scene 처리 완료. --- Output videos saved in", args.output_video_base_dir)
