"""Download/extract an artifact snapshot and validate the Jina-only runtime data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from retrieval_data import load_asr_metadata, load_retrieval_data  # noqa: E402


def extract_archives(data_dir: Path) -> None:
    archive_dir = data_dir / "archives"
    if not archive_dir.is_dir():
        return
    for archive_path in sorted(archive_dir.glob("*.zip")):
        marker = archive_path.with_suffix(archive_path.suffix + ".extracted")
        if marker.exists():
            continue
        print(f"Extracting {archive_path.name}...")
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (data_dir / member.filename).resolve()
                if data_dir.resolve() not in target.parents and target != data_dir.resolve():
                    raise ValueError(f"Unsafe ZIP member: {member.filename}")
            archive.extractall(data_dir)
        marker.touch()


def extract_ocr_metadata(data_dir: Path, manifest: dict) -> None:
    """Expand the transport ZIP into the directory consumed by the runtime."""
    paths = manifest["paths"]
    destination = data_dir / paths["ocr_metadata"]
    archive_path = data_dir / paths["ocr_metadata_archive"]
    if destination.is_dir():
        return
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Thiếu OCR metadata folder {destination} và archive {archive_path}"
        )

    print(f"Extracting {archive_path.name} -> {destination}...")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"Unsafe OCR ZIP member: {member.filename}")
        archive.extractall(destination)


def validate(data_dir: Path, manifest: dict, full: bool) -> None:
    expected_total = int(manifest["total_records"])
    expected_dim = int(manifest["embedding_dimension"])
    expected_collections = manifest["collections"]
    paths = {key: data_dir / value for key, value in manifest["paths"].items()}

    required_dirs = [
        "keyframes",
        "ocr_metadata",
        "asr_metadata",
        "jina_image_vectors",
        "jina_caption_vectors",
    ]
    missing = [str(paths[key]) for key in required_dirs if not paths[key].is_dir()]
    if missing:
        raise FileNotFoundError("Thiếu artifact:\n- " + "\n- ".join(missing))
    ocr_spec = manifest["ocr_metadata"]
    archive_path = paths["ocr_metadata_archive"]
    if archive_path.is_file():
        actual_size = archive_path.stat().st_size
        if actual_size != int(ocr_spec["size_bytes"]):
            raise ValueError(
                f"OCR ZIP có {actual_size} bytes, cần {ocr_spec['size_bytes']} bytes."
            )
        digest = hashlib.sha256()
        with archive_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != ocr_spec["sha256"]:
            raise ValueError("SHA256 của OCR metadata ZIP không khớp manifest.")

    retrieval = load_retrieval_data(paths["ocr_metadata"], paths["keyframes"])
    if len(retrieval.image_records) != expected_total:
        raise ValueError(
            f"Metadata có {len(retrieval.image_records):,} rows, cần {expected_total:,}."
        )
    blank_ocr = sum(
        not str(record.get("ocr_text") or "").strip()
        for record in retrieval.image_records
    )
    if blank_ocr != int(ocr_spec["blank_texts"]):
        raise ValueError(
            f"OCR metadata có {blank_ocr:,} text rỗng, "
            f"cần {int(ocr_spec['blank_texts']):,}."
        )

    for label, key in (
        ("Jina image", "jina_image_vectors"),
        ("Jina caption", "jina_caption_vectors"),
    ):
        offset = 0
        for collection, expected_rows in expected_collections.items():
            shard_path = paths[key] / f"{collection}.npy"
            if not shard_path.is_file():
                raise FileNotFoundError(f"Thiếu {shard_path}")
            shard = np.load(shard_path, mmap_mode="r")
            expected_shape = (int(expected_rows), expected_dim)
            if shard.shape != expected_shape or shard.dtype != np.float32:
                raise ValueError(
                    f"{shard_path}: nhận {shard.shape}/{shard.dtype}, "
                    f"cần {expected_shape}/float32."
                )
            end = offset + int(expected_rows)
            metadata_collections = {
                str(record["video_id"]).split("_", 1)[0]
                for record in retrieval.image_records[offset:end]
            }
            if metadata_collections != {collection}:
                raise ValueError(
                    f"{collection} không khớp metadata rows {offset}:{end}."
                )
            sample_rows = sorted({0, len(shard) // 2, len(shard) - 1})
            norms = np.linalg.norm(np.asarray(shard[sample_rows]), axis=1)
            if not np.allclose(norms, 1.0, atol=2e-3):
                raise ValueError(f"{shard_path} chưa L2-normalize: {norms.tolist()}")
            offset = end
        if offset != expected_total:
            raise ValueError(f"{label} có {offset:,} rows, cần {expected_total:,}.")
        print(f"OK {label}: {offset:,} vectors")

    _, _, asr_video_map = load_asr_metadata(paths["asr_metadata"])
    print(f"OK metadata: {expected_total:,} keyframes, {len(asr_video_map):,} ASR videos")

    print(
        f"OK OCR metadata: {len(retrieval.image_records):,} rows, "
        f"{blank_ocr:,} blank"
        + (", ZIP SHA256 matched" if archive_path.is_file() else "")
    )

    if full:
        missing_images = []
        for record in retrieval.image_records:
            relative_path = Path(record["path"].replace("Keyframes/", "", 1))
            if not (paths["keyframes"] / relative_path).is_file():
                missing_images.append(str(relative_path))
                if len(missing_images) >= 10:
                    break
        if missing_images:
            raise FileNotFoundError("Thiếu keyframe, ví dụ: " + ", ".join(missing_images))
        print(f"OK keyframe files: kiểm tra đủ {expected_total:,} metadata paths")

    if not paths["yolo_model"].is_file():
        print("WARNING: thiếu yolov8n.pt; app vẫn chạy nhưng Auto-Crop bị tắt.")
    print("Artifact validation PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--repo-id", help="Hugging Face dataset repo, vd user/aic-artifacts")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None, help="HF token; mặc định đọc HF_TOKEN")
    parser.add_argument("--full", action="store_true", help="Kiểm tra tồn tại từng keyframe")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.repo_id:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            token=args.token,
            local_dir=data_dir,
        )
    manifest = json.loads((REPO_ROOT / "artifacts-manifest.json").read_text("utf-8"))
    extract_archives(data_dir)
    extract_ocr_metadata(data_dir, manifest)
    validate(data_dir, manifest, args.full)


if __name__ == "__main__":
    main()
