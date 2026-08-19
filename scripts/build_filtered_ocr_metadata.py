"""Build a standalone metadata ZIP with filtered OCR text embedded in each row."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from retrieval_data import load_retrieval_data, overlay_ocr_jsonl  # noqa: E402


def build_filtered_metadata(
    metadata_dir: Path,
    ocr_text_dir: Path,
    keyframes_dir: Path,
    output_zip: Path,
) -> None:
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy metadata source: {metadata_dir}")

    retrieval = load_retrieval_data(metadata_dir, keyframes_dir)
    # Cho phép rebuild riêng L21/L22 trên nền artifact hiện tại; mọi collection
    # không có JSONL trong input được giữ nguyên ocr_text từ metadata source.
    stats = overlay_ocr_jsonl(
        ocr_text_dir,
        retrieval.image_records,
        require_all_records=False,
    )
    text_by_index = {
        int(record["idx"]): str(record.get("ocr_text") or "")
        for record in retrieval.image_records
    }

    json_paths = sorted(metadata_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"Không có JSON trong {metadata_dir}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = output_zip.with_suffix(output_zip.suffix + ".tmp")
    written_indices: set[int] = set()
    compression = zipfile.ZIP_DEFLATED

    try:
        with zipfile.ZipFile(
            temporary_zip,
            mode="w",
            compression=compression,
            compresslevel=6,
        ) as archive:
            for json_path in json_paths:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                if not isinstance(payload, list):
                    raise ValueError(f"Metadata không phải JSON array: {json_path}")

                for record in payload:
                    record_index = int(record["idx"])
                    if record_index in written_indices:
                        raise ValueError(f"Trùng idx={record_index} trong metadata source")
                    if record_index not in text_by_index:
                        raise ValueError(f"Không có OCR text cho idx={record_index}")
                    record["ocr_text"] = text_by_index[record_index]
                    written_indices.add(record_index)

                relative_name = json_path.relative_to(metadata_dir).as_posix()
                archive.writestr(
                    relative_name,
                    json.dumps(payload, ensure_ascii=False, indent=2),
                )

        expected_indices = set(text_by_index)
        if written_indices != expected_indices:
            missing = sorted(expected_indices - written_indices)
            raise ValueError(
                f"Output thiếu {len(missing)} idx; ví dụ {missing[:5]}"
            )

        os.replace(temporary_zip, output_zip)
    except Exception:
        if temporary_zip.exists():
            temporary_zip.unlink()
        raise

    # Reload through the same production loader, then compare every OCR string.
    validated = load_retrieval_data(
        output_zip,
        keyframes_dir,
        expected_rows=len(retrieval.image_records),
    )
    mismatches = [
        index
        for index, record in enumerate(validated.image_records)
        if record.get("ocr_text", "") != retrieval.image_records[index].get("ocr_text", "")
    ]
    if mismatches:
        raise ValueError(f"OCR text không khớp sau khi reload ZIP: {mismatches[:5]}")

    print(f"Đã tạo: {output_zip}")
    print(f"Kích thước: {output_zip.stat().st_size / (1024 ** 2):.2f} MiB")
    print(f"Metadata JSON: {len(json_paths):,} files")
    print(f"Records: {len(validated.image_records):,}")
    total_blank_texts = sum(
        not record.get("ocr_text", "").strip()
        for record in validated.image_records
    )
    print(f"OCR rỗng toàn bộ metadata: {total_blank_texts:,}")
    print(
        "Đã lọc L21/L22: "
        f"{stats['filtered_ticker_lines']:,} ticker lines, "
        f"{stats['removed_text_segments']:,} text segments"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=REPO_ROOT / "ocr" / "metadata_ocr_filtered",
    )
    parser.add_argument(
        "--ocr-text-dir",
        type=Path,
        default=REPO_ROOT / "OCR_original_no_LLM" / "OCR",
    )
    parser.add_argument(
        "--keyframes-dir",
        type=Path,
        default=REPO_ROOT / "keyframes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "ocr" / "metadata_ocr_filtered.zip",
    )
    args = parser.parse_args()
    build_filtered_metadata(
        args.metadata_dir.resolve(),
        args.ocr_text_dir.resolve(),
        args.keyframes_dir.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
