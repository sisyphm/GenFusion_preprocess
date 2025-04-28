#!/usr/bin/env python3
import zipfile
import shutil
import logging
from pathlib import Path

#---------------- Configuration ----------------#
SRC_ROOT = Path("/home/ubuntu/DL3DV-ALL-960P")       # DL3DV-ALL-960P 최상위 경로
DST_ROOT = Path("/home/ubuntu/DL3DV-ALL-ColmapCache")  # DL3DV-ALL-ColmapCache 최상위 경로
LOG_FILE = Path("transfer_images4.log")            # 로그 파일 경로
#-----------------------------------------------#

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def extract_and_flatten(zip_path: Path, extract_dir: Path) -> Path:
    """
    zip_path이 존재하면 extract_dir에 압축 해제.
    이미 extract_dir이 있으면 그대로 사용.
    내부에 중첩된 scene 폴더가 있으면 평탄화.
    zip 파일은 추출 성공 후에만 삭제하도록 별도 처리.
    """
    if extract_dir.exists():
        logging.info(f"Directory exists, skipping extraction: {extract_dir}")
        return extract_dir
    if zip_path.exists():
        logging.info(f"Extracting {zip_path} → {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        # 중첩된 scene 폴더 평탄화
        nested = extract_dir / extract_dir.name
        if nested.exists() and nested.is_dir():
            logging.info(f"Flattening nested directory {nested}")
            for child in nested.iterdir():
                dest = extract_dir / child.name
                if dest.exists():
                    if dest.is_dir(): shutil.rmtree(dest)
                    else: dest.unlink()
                child.replace(dest)
            nested.rmdir()
        return extract_dir
    return None


def flatten_dst_dir(dst_dir: Path, scene: str):
    """
    DST extract 디렉토리에서 중첩된 scene 폴더가 있으면 평탄화.
    """
    nested = dst_dir / scene
    if nested.exists() and nested.is_dir():
        logging.info(f"Flattening nested DST directory {nested}")
        for child in nested.iterdir():
            dest = dst_dir / child.name
            if dest.exists():
                if dest.is_dir(): shutil.rmtree(dest)
                else: dest.unlink()
            child.replace(dest)
        nested.rmdir()


def find_images4_folder(extract_dir: Path) -> Path:
    """
    extract_dir 하위에서 images_4 폴더를 재귀 탐색하여 반환.
    """
    candidates = list(extract_dir.rglob("images_4"))
    if not candidates:
        return None
    if len(candidates) > 1:
        logging.warning(f"Multiple 'images_4' folders found, using first: {candidates[0]}")
    return candidates[0]


def process_scene(res_dir: Path, scene: str):
    """
    하나의 씬을 처리:
      1) SRC zip 추출 및 평탄화
      2) images_4 폴더 탐색
      3) DST zip 추출 및 평탄화
      4) images_4 이동
      5) 성공 시 zip 및 임시 폴더 정리
    """
    try:
        # 1) SRC 준비
        src_zip = res_dir / f"{scene}.zip"
        src_extract = res_dir / scene
        src_dir = extract_and_flatten(src_zip, src_extract)
        if not src_dir:
            logging.error(f"Source not found for {scene}, skipping.")
            return

        # 2) images_4 탐색
        src_images = find_images4_folder(src_dir)
        if not src_images or not src_images.is_dir():
            logging.error(f"Missing images_4 in {src_dir}, will retry later.")
            shutil.rmtree(src_dir, ignore_errors=True)
            return

        # 3) DST 준비
        dst_res_dir = DST_ROOT / res_dir.name
        dst_zip = dst_res_dir / f"{scene}.zip"
        dst_extract = dst_res_dir / scene
        dst_dir = extract_and_flatten(dst_zip, dst_extract)
        if dst_dir:
            flatten_dst_dir(dst_dir, scene)
        else:
            dst_extract.mkdir(parents=True, exist_ok=True)
            dst_dir = dst_extract

        # 4) images_4 이동
        dst_colmap = dst_dir / "colmap"
        dst_images = dst_colmap / "images_4"
        dst_colmap.mkdir(parents=True, exist_ok=True)
        if dst_images.exists():
            logging.warning(f"Destination exists, removing: {dst_images}")
            shutil.rmtree(dst_images)
        logging.info(f"Moving {src_images} → {dst_images}")
        shutil.move(str(src_images), str(dst_images))

        # 5) 성공 시 정리
        # SRC zip 삭제
        if src_zip.exists():
            logging.info(f"Deleting source zip {src_zip}")
            src_zip.unlink()
        # SRC extract 폴더 삭제
        shutil.rmtree(src_dir)
        # DST zip 삭제
        if dst_zip.exists():
            logging.info(f"Deleting DST zip {dst_zip}")
            dst_zip.unlink()

        logging.info(f"Completed scene {scene} in {res_dir.name}")
    except Exception as e:
        logging.exception(f"Error processing {scene}: {e}")


def main():
    setup_logging()
    logging.info("=== Start processing all scenes ===")

    for res_dir in sorted(SRC_ROOT.iterdir()):
        if not res_dir.is_dir() or res_dir.name == ".cache":
            continue
        scenes = set()
        scenes.update(p.stem for p in res_dir.glob("*.zip"))
        scenes.update(d.name for d in res_dir.iterdir() if d.is_dir() and d.name != ".cache")
        for scene in sorted(scenes):
            process_scene(res_dir, scene)

    logging.info("=== All done ===")

if __name__ == "__main__":
    main()
