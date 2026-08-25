"""Load the AIC 2026 retrieval metadata without coupling it to Flask.

The current dataset stores OCR-enriched keyframe metadata in a zip archive.
This module normalizes that metadata to the field names used by ``app.py`` and
keeps the web paths independent from the machine's absolute filesystem path.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterator
import zipfile


VIDEO_ID_PATTERN = re.compile(r"^[A-Z]\d{2}_V\d+$", re.IGNORECASE)
OCR_OVERLAY_FILTER_COLLECTIONS = {"L21", "L22"}
# L21/L22 dùng layout bản tin 1280x720. Ticker nhiễu là dải ngang sát đáy,
# vì vậy chỉ xét trục Y và mặc định phủ toàn bộ chiều rộng frame.
OCR_BOTTOM_TICKER_Y_MIN = 640.0


@dataclass(slots=True)
class RetrievalData:
    image_records: list[dict]
    metadata_cache: dict[str, dict[int, dict]]
    keyframe_time_cache: dict[str, dict]
    video_frame_ids: dict[str, list[int]]
    video_url_cache: dict[str, str]


def parse_keyframe_path(original_path: str | Path | None):
    """Return ``(web_path, video_id, frame_id)`` for supported keyframe paths.

    Accepted inputs include the paths stored in ``metadata_ocr.zip``, paths
    already returned to the browser, absolute Windows paths, and the legacy
    ``Keyframes_Lxx`` layout.
    """

    if original_path is None:
        return None, "N/A", None

    path_text = str(original_path).strip().replace("\\", "/")
    path_text = path_text.split("?", 1)[0].split("#", 1)[0]
    parts = [part for part in PurePosixPath(path_text).parts if part not in {"/", ""}]

    video_position = next(
        (position for position, part in enumerate(parts) if VIDEO_ID_PATTERN.fullmatch(part)),
        None,
    )
    if video_position is None or video_position + 1 >= len(parts):
        return None, "N/A", None

    video_id = parts[video_position].upper()
    image_name = parts[-1]
    image_path = PurePosixPath(image_name)
    if not image_path.stem.isdigit() or not image_path.suffix:
        return None, "N/A", None

    collection = video_id.split("_", 1)[0]
    web_path = f"Keyframes/{collection}/{video_id}/{image_name}"
    return web_path, video_id, image_path.stem


def _iter_metadata_documents(metadata_source: Path) -> Iterator[list[dict]]:
    if metadata_source.is_file() and metadata_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(metadata_source) as archive:
            json_names = sorted(
                name for name in archive.namelist() if name.lower().endswith(".json")
            )
            if not json_names:
                raise ValueError(f"Archive metadata không chứa file JSON: {metadata_source}")
            for name in json_names:
                yield json.loads(archive.read(name).decode("utf-8"))
        return

    if metadata_source.is_dir():
        json_paths = sorted(metadata_source.rglob("*.json"))
        if not json_paths:
            raise ValueError(f"Thư mục metadata không chứa file JSON: {metadata_source}")
        for json_path in json_paths:
            with json_path.open("r", encoding="utf-8") as stream:
                yield json.load(stream)
        return

    raise FileNotFoundError(f"Không tìm thấy metadata OCR: {metadata_source}")


def _number(record: dict, *keys: str, default=0):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def load_retrieval_data(
    metadata_source: str | Path,
    keyframes_dir: str | Path,
    expected_rows: int | None = None,
) -> RetrievalData:
    """Load OCR/keyframe metadata and build the lookup tables used by Flask."""

    metadata_source = Path(metadata_source)
    keyframes_dir = Path(keyframes_dir)
    if not keyframes_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục keyframe: {keyframes_dir}")

    image_records: list[dict | None] = []
    metadata_cache: dict[str, dict[int, dict]] = {}
    video_url_cache: dict[str, str] = {}

    for document in _iter_metadata_documents(metadata_source):
        if not isinstance(document, list):
            raise ValueError("Mỗi file metadata keyframe phải chứa một JSON array.")

        for source_record in document:
            video_id = str(source_record.get("video_id", "")).upper()
            if not VIDEO_ID_PATTERN.fullmatch(video_id):
                raise ValueError(f"video_id không hợp lệ trong metadata: {video_id!r}")

            raw_path = source_record.get("path")
            web_path, parsed_video_id, parsed_frame_id = parse_keyframe_path(raw_path)
            if web_path is None or parsed_video_id != video_id:
                raise ValueError(f"Đường dẫn keyframe không hợp lệ: {raw_path!r}")

            frame_id = int(_number(source_record, "frame_id", "frame_idx", "n", default=parsed_frame_id))
            frame_idx = int(_number(source_record, "frame_idx", "frame_id", default=frame_id))
            pts_time = float(_number(source_record, "pts_time", "frame_stamp", default=0.0))
            record_index = int(_number(source_record, "idx", "keyframe_order", default=len(image_records)))

            record = dict(source_record)
            record.update(
                {
                    "path": web_path,
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "frame_idx": frame_idx,
                    "pts_time": pts_time,
                    "ocr_text": source_record.get("ocr_text", "") or "",
                }
            )

            if record_index >= len(image_records):
                image_records.extend([None] * (record_index + 1 - len(image_records)))
            if image_records[record_index] is not None:
                raise ValueError(f"Trùng idx={record_index} trong metadata OCR.")
            image_records[record_index] = record

            video_records = metadata_cache.setdefault(video_id, {})
            if frame_id in video_records:
                raise ValueError(f"Trùng frame_id={frame_id} của video {video_id}.")
            video_records[frame_id] = record

            video_url = source_record.get("video_url")
            if video_url:
                video_url_cache[video_id] = str(video_url)

    missing_indices = [index for index, record in enumerate(image_records) if record is None]
    if missing_indices:
        preview = ", ".join(map(str, missing_indices[:5]))
        raise ValueError(f"Metadata OCR thiếu idx: {preview}")

    if expected_rows is not None and len(image_records) != expected_rows:
        raise ValueError(
            "Số metadata không khớp FAISS index: "
            f"metadata={len(image_records)}, faiss={expected_rows}"
        )

    normalized_records: list[dict] = [record for record in image_records if record is not None]
    keyframe_time_cache: dict[str, dict] = {}
    video_frame_ids: dict[str, list[int]] = {}

    for video_id, frame_records in metadata_cache.items():
        ordered = sorted(
            frame_records.items(),
            key=lambda item: (float(item[1]["pts_time"]), item[0]),
        )
        video_frame_ids[video_id] = [frame_id for frame_id, _ in ordered]
        keyframe_time_cache[video_id] = {
            "times": [float(record["pts_time"]) for _, record in ordered],
            "data": [
                (frame_id, int(record["frame_idx"]))
                for frame_id, record in ordered
            ],
            "paths": [str(record["path"]) for _, record in ordered],
            "fps": next(
                (float(record["fps"]) for _, record in ordered if record.get("fps") is not None),
                None,
            ),
        }

    return RetrievalData(
        image_records=normalized_records,
        metadata_cache=metadata_cache,
        keyframe_time_cache=keyframe_time_cache,
        video_frame_ids=video_frame_ids,
        video_url_cache=video_url_cache,
    )


def load_asr_metadata(asr_dir: str | Path):
    """Load both the current list-based ASR schema and the legacy schema."""

    asr_dir = Path(asr_dir)
    if not asr_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ASR: {asr_dir}")

    asr_video_map: dict[str, list[dict]] = {}
    for json_path in sorted(asr_dir.rglob("*.json")):
        with json_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)

        source_segments = payload.get("segments", []) if isinstance(payload, dict) else payload
        if not isinstance(source_segments, list):
            raise ValueError(f"ASR JSON không đúng schema: {json_path}")

        for source_segment in source_segments:
            text = str(source_segment.get("text", "")).strip()
            if not text:
                continue
            video_id = str(source_segment.get("video_id") or json_path.stem).upper()
            segment = {
                "video_id": video_id,
                "text": text,
                "start": float(_number(source_segment, "start", "t_start", default=0.0)),
                "end": float(_number(source_segment, "end", "t_end", default=0.0)),
            }
            for optional_key in ("id", "frame_start", "frame_end"):
                if optional_key in source_segment:
                    segment[optional_key] = source_segment[optional_key]
            asr_video_map.setdefault(video_id, []).append(segment)

    asr_data: list[dict] = []
    asr_corpus_tokenized: list[list[str]] = []
    for video_id in sorted(asr_video_map):
        segments = asr_video_map[video_id]
        segments.sort(key=lambda segment: (segment["start"], segment["end"]))
        for segment_index, segment in enumerate(segments):
            segment["video_segment_index"] = segment_index
            asr_data.append(segment)
            asr_corpus_tokenized.append(segment["text"].lower().split())

    return asr_data, asr_corpus_tokenized, asr_video_map


def overlay_ocr_jsonl(
    ocr_dir: str | Path,
    image_records: list[dict],
    require_all_records: bool = True,
) -> dict[str, int]:
    """Replace ``ocr_text`` using collection-level OCR JSONL files.

    The JSONL dataset only contains ``video``, ``kf`` and OCR fields; it does
    not contain timestamps or video URLs. Therefore it is overlaid onto the
    canonical records loaded from keyframe metadata instead of replacing that
    metadata source entirely. Every keyframe must occur exactly once so BM25
    row indices remain aligned with the semantic indexes.
    """

    ocr_dir = Path(ocr_dir)
    if not ocr_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục OCR JSONL: {ocr_dir}")

    jsonl_paths = sorted(ocr_dir.glob("L*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"Không có shard OCR L*.jsonl trong {ocr_dir}")

    record_by_key: dict[tuple[str, int], dict] = {}
    for record in image_records:
        key = (str(record.get("video_id", "")).upper(), int(record["frame_id"]))
        if key in record_by_key:
            raise ValueError(f"Metadata keyframe trùng khóa OCR: {key}")
        record_by_key[key] = record

    seen: set[tuple[str, int]] = set()
    source_collections = {path.stem.upper() for path in jsonl_paths}
    blank_texts = 0
    filtered_ticker_lines = 0
    removed_text_segments = 0
    for jsonl_path in jsonl_paths:
        with jsonl_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    source_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON OCR lỗi tại {jsonl_path.name}:{line_number}: {exc}"
                    ) from exc

                video_id = str(source_record.get("video", "")).upper()
                frame_text = str(source_record.get("kf", "")).strip()
                if not VIDEO_ID_PATTERN.fullmatch(video_id) or not frame_text.isdigit():
                    raise ValueError(
                        f"Khóa OCR không hợp lệ tại {jsonl_path.name}:{line_number}: "
                        f"video={video_id!r}, kf={frame_text!r}"
                    )
                key = (video_id, int(frame_text))
                if key in seen:
                    raise ValueError(f"OCR JSONL trùng keyframe {key}")
                target_record = record_by_key.get(key)
                if target_record is None:
                    raise ValueError(
                        f"OCR JSONL có keyframe ngoài canonical metadata: {key}"
                    )

                text = str(source_record.get("text") or "").strip()
                collection = video_id.split("_", 1)[0]
                if collection in OCR_OVERLAY_FILTER_COLLECTIONS:
                    filtered_segments: Counter[str] = Counter()
                    for ocr_line in source_record.get("lines", []):
                        line_text = str(ocr_line.get("text") or "").strip()
                        bbox = ocr_line.get("bbox")
                        if not line_text or not (
                            isinstance(bbox, list) and len(bbox) == 4
                        ):
                            continue

                        y1, y2 = float(bbox[1]), float(bbox[3])
                        center_y = (y1 + y2) / 2.0
                        # Full-width bottom band: không xét x1/x2, nên ticker
                        # vẫn bị loại khi OCR chia thành nhiều bbox ngắn.
                        is_bottom_ticker = center_y >= OCR_BOTTOM_TICKER_Y_MIN
                        if is_bottom_ticker:
                            filtered_segments[line_text] += 1
                            filtered_ticker_lines += 1

                    if filtered_segments and text:
                        kept_segments = []
                        for segment in (part.strip() for part in text.split(" | ")):
                            if filtered_segments[segment] > 0:
                                filtered_segments[segment] -= 1
                                removed_text_segments += 1
                            elif segment:
                                kept_segments.append(segment)
                        text = " | ".join(kept_segments)

                target_record["ocr_text"] = text
                if not text:
                    blank_texts += 1
                seen.add(key)

    if require_all_records:
        required_keys = set(record_by_key)
    else:
        required_keys = {
            key
            for key in record_by_key
            if key[0].split("_", 1)[0] in source_collections
        }
    missing = required_keys - seen
    if missing:
        preview = ", ".join(f"{video}/{frame:06d}" for video, frame in sorted(missing)[:5])
        raise ValueError(
            f"OCR JSONL thiếu {len(missing)} keyframe canonical; ví dụ: {preview}"
        )

    return {
        "files": len(jsonl_paths),
        "rows": len(seen),
        "blank_texts": blank_texts,
        "filtered_ticker_lines": filtered_ticker_lines,
        "removed_text_segments": removed_text_segments,
    }
