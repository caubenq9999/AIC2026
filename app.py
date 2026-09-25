# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import os
from pathlib import Path
from functools import lru_cache
from html import escape as html_escape

# Mọi đường dẫn mặc định đều neo theo vị trí app.py, không phụ thuộc cwd.
BASE_DIR = Path(__file__).resolve().parent


def project_path(environment_variable, *relative_parts):
    configured_path = os.getenv(environment_variable)
    if configured_path:
        configured_path = Path(configured_path).expanduser()
        if not configured_path.is_absolute():
            configured_path = BASE_DIR / configured_path
        return configured_path.resolve()
    return BASE_DIR.joinpath(*relative_parts).resolve()


ARTIFACTS_DIR = project_path("AIC_ARTIFACTS_DIR", "artifacts")


def canonical_or_legacy_path(environment_variable, canonical_parts, legacy_parts):
    """Prefer the unified artifact root and retain old layouts during migration."""
    if os.getenv(environment_variable, "").strip():
        return project_path(environment_variable)
    canonical = ARTIFACTS_DIR.joinpath(*canonical_parts).resolve()
    if canonical.exists():
        return canonical
    return BASE_DIR.joinpath(*legacy_parts).resolve()


def resolve_ocr_metadata_path():
    """Resolve frame metadata with embedded OCR from canonical or legacy paths."""
    configured_path = os.getenv("AIC_OCR_METADATA_PATH", "").strip()
    if configured_path:
        return project_path("AIC_OCR_METADATA_PATH")

    candidates = (
        ARTIFACTS_DIR / "metadata" / "frames",
        BASE_DIR / "metadata" / "frames",
        BASE_DIR / "ocr" / "metadata_ocr_filtered",
        BASE_DIR / "ocr" / "metadata_ocr_filtered.zip",
        BASE_DIR / "ocr" / "metadata_ocr",
        BASE_DIR / "ocr" / "metadata_ocr.zip",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        # Batch-1 ShardedNpyIndex still needs the L21-L30 row order. The new
        # root metadata/frames may currently contain only M/N/S, so do not
        # select it as the L metadata source merely because the folder exists.
        if candidate.is_dir() and next(candidate.rglob("L21_V*.json"), None) is not None:
            return candidate.resolve()

    # Keep the error emitted by load_retrieval_data deterministic and useful.
    return candidates[0].resolve()


# Có thể đổi vị trí cache bằng AIC_CACHE_DIR mà không cần sửa source.
CACHE_DIR = project_path("AIC_CACHE_DIR", ".cache", "huggingface")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["TORCH_HOME"] = str(CACHE_DIR)
os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR)

import torch
import requests
import numpy as np
import json
import csv
import zipfile
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_from_directory, send_file, g, abort, Response
from dres_gateway import check_evaluations, submit_answer as submit_to_dres
import gc
from retrieval_data import (
    load_asr_metadata,
    load_retrieval_data,
    overlay_ocr_jsonl,
    parse_keyframe_path,
)
from semantic_search import (
    JinaTextEncoder,
    ModelUnavailableError,
    ShardedNpyIndex,
)
from portable_image_search import (
    PortableJinaImageIndex,
    normalized_frame_key,
    video_id_aliases,
)
from search_bm25 import PersistentInvertedBM25, fingerprint_paths
from traffic_search import (
    TrafficSearchIndex,
    expanded_query,
    merge_ranked_batches,
    parse_traffic_query,
)
from groq import Groq
from flask_cors import CORS
# from rank_bm25 import BM25Okapi # <-- XÓA BỎ (Không dùng thư viện nữa)
import re 
import collections
import math 
import bisect 
from PIL import Image # (THÊM MỚI) Thêm PIL để xử lý ảnh upload
import io # (THÊM MỚI) Thêm io
from ultralytics import YOLO # (THÊM MỚI) YOLOv8 cho auto-crop pre-processing


print("--- KHỞI ĐỘNG HỆ THỐNG TRUY VẤN HÌNH ẢNH ---")

# --- 2. CẤU HÌNH ---
index_name = "aic_ocr_index"
KEYFRAMES_DIR = canonical_or_legacy_path(
    "AIC_KEYFRAMES_DIR", ("keyframes",), ("keyframes",)
)
OCR_METADATA_PATH = resolve_ocr_metadata_path()
OCR_TEXT_DIR = project_path(
    "AIC_OCR_TEXT_DIR", "OCR_original_no_LLM", "OCR"
)
if os.getenv("AIC_ASR_METADATA_DIR", "").strip():
    ASR_METADATA_DIR = project_path("AIC_ASR_METADATA_DIR")
else:
    ASR_METADATA_DIR = next(
        (
            path.resolve()
            for path in (
                ARTIFACTS_DIR / "metadata" / "asr",
                BASE_DIR / "metadata" / "asr",
                BASE_DIR / "asr" / "metadata_asr_clean",
            )
            if path.exists()
        ),
        (ARTIFACTS_DIR / "metadata" / "asr").resolve(),
    )
YOLO_MODEL_PATH = project_path("AIC_YOLO_MODEL_PATH", "yolov8n.pt")


def resolve_unified_jina_root():
    """Return the standardized Jina root, or None while using legacy paths."""
    if os.getenv("AIC_JINA_EMBEDDINGS_DIR", "").strip():
        return project_path("AIC_JINA_EMBEDDINGS_DIR")
    # Layout đang dùng trong bộ artifact của đội là ``artifacts/embedding``
    # (số ít). Vẫn dò ``embeddings`` để các máy đã tải layout tài liệu cũ
    # không bị hỏng sau thay đổi này.
    for candidate in (
        ARTIFACTS_DIR / "embedding" / "jina",
        ARTIFACTS_DIR / "embeddings" / "jina",
    ):
        if candidate.is_dir():
            return candidate.resolve()
    return None


UNIFIED_JINA_ROOT = resolve_unified_jina_root()
if os.getenv("AIC_JINA_VECTORS_DIR", "").strip():
    JINA_VECTORS_DIR = project_path("AIC_JINA_VECTORS_DIR")
elif UNIFIED_JINA_ROOT is not None and (UNIFIED_JINA_ROOT / "image").is_dir():
    JINA_VECTORS_DIR = (UNIFIED_JINA_ROOT / "image").resolve()
else:
    standardized = BASE_DIR / "embedding" / "jina" / "image"
    JINA_VECTORS_DIR = (
        standardized.resolve()
        if standardized.is_dir()
        else (BASE_DIR / "embedding" / "jina" / "jina_embeddings_npy").resolve()
    )

if os.getenv("AIC_JINA_CAPTION_VECTORS_DIR", "").strip():
    JINA_CAPTION_VECTORS_DIR = project_path("AIC_JINA_CAPTION_VECTORS_DIR")
elif UNIFIED_JINA_ROOT is not None and (UNIFIED_JINA_ROOT / "caption").is_dir():
    JINA_CAPTION_VECTORS_DIR = (UNIFIED_JINA_ROOT / "caption").resolve()
else:
    standardized = BASE_DIR / "embedding" / "jina" / "caption"
    JINA_CAPTION_VECTORS_DIR = (
        standardized.resolve()
        if standardized.is_dir()
        else (BASE_DIR / "embedding" / "jina" / "caption_embeddings_npy").resolve()
    )
SEARCH_INDEX_CACHE_DIR = project_path(
    "AIC_SEARCH_INDEX_CACHE_DIR", ".cache", "search_indices"
)
def resolve_videos_dir():
    """Use the new ``videos`` folder while preserving the legacy ``video`` path."""
    configured = os.getenv("AIC_VIDEOS_DIR", "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = BASE_DIR / configured_path
        return configured_path.resolve()

    canonical = (ARTIFACTS_DIR / "videos").resolve()
    if canonical.is_dir():
        return canonical
    preferred = (BASE_DIR / "videos").resolve()
    legacy = (BASE_DIR / "video").resolve()
    if preferred.is_dir() or not legacy.is_dir():
        return preferred
    return legacy


# Video local là artifact tùy chọn. Nếu không có MP4 đã giải nén,
# build_playback_info() sẽ tự fallback về URL YouTube trong metadata.
VIDEOS_DIR = resolve_videos_dir()
TRAFFIC_CAPTION_DIR = (
    project_path("AIC_TRAFFIC_CAPTION_DIR")
    if os.getenv("AIC_TRAFFIC_CAPTION_DIR", "").strip()
    else (
        (UNIFIED_JINA_ROOT / "caption").resolve()
        if (
            UNIFIED_JINA_ROOT is not None
            and (UNIFIED_JINA_ROOT / "caption").is_dir()
        )
        else (
            (BASE_DIR / "embedding" / "jina" / "caption").resolve()
            if (BASE_DIR / "embedding" / "jina" / "caption").is_dir()
            else (BASE_DIR / "captionbatch2_emb").resolve()
        )
    )
)
if os.getenv("AIC_TRAFFIC_DETECTION_PATH", "").strip():
    TRAFFIC_DETECTION_PATH = project_path("AIC_TRAFFIC_DETECTION_PATH")
else:
    TRAFFIC_DETECTION_PATH = next(
        (
            path.resolve()
            for path in (
                ARTIFACTS_DIR / "detections",
                BASE_DIR / "detections",
                BASE_DIR / "detection segmentation" / "detection segmentation",
            )
            if path.exists()
        ),
        (ARTIFACTS_DIR / "detections").resolve(),
    )
TRAFFIC_KEYFRAMES_DIR = canonical_or_legacy_path(
    "AIC_TRAFFIC_KEYFRAMES_DIR",
    ("keyframes",),
    ("keyframes",),
)
TRAFFIC_MAP_DIR = canonical_or_legacy_path(
    "AIC_TRAFFIC_MAP_DIR",
    ("keyframes",),
    ("keyframes",),
)
# Chỉ còn là fallback legacy cho vài video chưa được chép video_url vào metadata.
# Cấu trúc artifact chuẩn không có media-info riêng.
TRAFFIC_MEDIA_INFO_DIR = project_path(
    "AIC_TRAFFIC_MEDIA_INFO_DIR", "aic26-b2-media-info", "media-info"
)
if os.getenv("AIC_TRAFFIC_METADATA_DIR", "").strip():
    TRAFFIC_METADATA_DIR = project_path("AIC_TRAFFIC_METADATA_DIR")
else:
    TRAFFIC_METADATA_DIR = next(
        (
            path.resolve()
            for path in (
                ARTIFACTS_DIR / "metadata" / "frames",
                BASE_DIR / "metadata" / "frames",
                BASE_DIR / "ocr" / "metadata_ocr_filtered" / "metadata",
            )
            if path.exists()
        ),
        (ARTIFACTS_DIR / "metadata" / "frames").resolve(),
    )
if os.getenv("AIC_JINA_PORTABLE_IMAGE_DIR", "").strip():
    JINA_PORTABLE_IMAGE_DIR = project_path("AIC_JINA_PORTABLE_IMAGE_DIR")
elif UNIFIED_JINA_ROOT is not None and (UNIFIED_JINA_ROOT / "image").is_dir():
    JINA_PORTABLE_IMAGE_DIR = (UNIFIED_JINA_ROOT / "image").resolve()
else:
    JINA_PORTABLE_IMAGE_DIR = (BASE_DIR / "embedding" / "jina" / "image").resolve()
TRAFFIC_SEARCH_CACHE_DIR = project_path(
    "AIC_TRAFFIC_SEARCH_CACHE_DIR", ".cache", "batch2_search"
)
TRAFFIC_SEARCH_DIMS = max(
    32, min(int(os.getenv("AIC_TRAFFIC_SEARCH_DIMS", "128")), 1024)
)
SEMANTIC_QUERY_CACHE_SIZE = max(
    0, int(os.getenv("AIC_SEMANTIC_QUERY_CACHE_SIZE", "256"))
)
SEMANTIC_RESULT_CACHE_SIZE = max(
    0, int(os.getenv("AIC_SEMANTIC_RESULT_CACHE_SIZE", "128"))
)
LOCAL_VIDEO_ID_PATTERN = re.compile(r"^[A-Z]\d{2,3}[-_]V\d+$", re.IGNORECASE)
BROWSER_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v"}


def build_local_video_index(videos_dir):
    """Index extracted videos and MP4 members kept inside downloaded ZIPs."""
    videos_dir = Path(videos_dir)
    if not videos_dir.is_dir():
        print(f"CẢNH BÁO: Không tìm thấy folder video local {videos_dir}.")
        return {}

    index = {}
    # File đã giải nén được ưu tiên nếu cùng video cũng xuất hiện trong ZIP.
    for video_path in sorted(videos_dir.rglob("*")):
        if (
            not video_path.is_file()
            or video_path.suffix.lower() not in BROWSER_VIDEO_EXTENSIONS
        ):
            continue
        video_id = video_path.stem.upper()
        if not LOCAL_VIDEO_ID_PATTERN.fullmatch(video_id):
            continue
        resolved_path = video_path.resolve()
        if video_id in index:
            print(
                f"CẢNH BÁO: Trùng video local {video_id}; "
                f"giữ {index[video_id]}, bỏ qua {resolved_path}."
            )
            continue
        index[video_id] = {
            "kind": "file",
            "path": resolved_path,
            "size": resolved_path.stat().st_size,
        }

    for archive_path in sorted(videos_dir.rglob("*.zip")):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    member_path = Path(member.filename)
                    if member_path.suffix.lower() not in BROWSER_VIDEO_EXTENSIONS:
                        continue
                    video_id = member_path.stem.upper()
                    if not LOCAL_VIDEO_ID_PATTERN.fullmatch(video_id):
                        continue
                    if video_id in index:
                        continue
                    index[video_id] = {
                        "kind": "zip",
                        "path": archive_path.resolve(),
                        "member": member.filename,
                        "size": int(member.file_size),
                    }
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"CẢNH BÁO: Không đọc được ZIP video {archive_path}: {exc}")
    return index


local_video_index = build_local_video_index(VIDEOS_DIR)
print(f"Loaded {len(local_video_index)} local videos from {VIDEOS_DIR}.")

# --- 4. CẤU HÌNH GROQ API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = "openai/gpt-oss-120b"
try:
    if not GROQ_API_KEY:
        raise ValueError("Chưa cấu hình biến môi trường GROQ_API_KEY.")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Kết nối với Groq API thành công.")
except Exception as e:
    print(f"Lỗi khi cấu hình Groq API: {e}")
    groq_client = None

def groq_generate(prompt):
    """Gọi Groq chat completions (model GROQ_MODEL), trả về text output."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# --- 5. TẢI CÁC MODEL VÀ DỮ LIỆU ---
print("Đang tải model và các tài nguyên, vui lòng đợi...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# (THÊM MỚI) Tải YOLOv8n model cho auto-crop pre-processing
yolo_model = None
try:
    yolo_model = YOLO(str(YOLO_MODEL_PATH))  # Tự động tải về nếu chưa có
    yolo_model.to(device)
    print(f"Model YOLOv8n đã được tải thành công lên thiết bị: {device.upper()}")
except Exception as e:
    print(f"CẢNH BÁO: Không thể tải YOLOv8n model: {e}. Tính năng auto_crop sẽ bị vô hiệu hóa.")
    yolo_model = None

retrieval_data = load_retrieval_data(
    OCR_METADATA_PATH,
    KEYFRAMES_DIR,
    allowed_collections={f"L{number}" for number in range(21, 31)},
)
image_records = retrieval_data.image_records
metadata_cache = retrieval_data.metadata_cache
keyframe_time_cache = retrieval_data.keyframe_time_cache
video_frame_ids = retrieval_data.video_frame_ids
video_url_cache = retrieval_data.video_url_cache
print(f"Loaded {len(image_records)} embedding/OCR records from {len(metadata_cache)} videos.")

traffic_search_index = None
traffic_search_reason = None
try:
    traffic_search_index = TrafficSearchIndex(
        caption_dir=TRAFFIC_CAPTION_DIR,
        detection_path=TRAFFIC_DETECTION_PATH,
        keyframes_dir=TRAFFIC_KEYFRAMES_DIR,
        map_dir=TRAFFIC_MAP_DIR,
        media_info_dir=TRAFFIC_MEDIA_INFO_DIR,
        metadata_dir=TRAFFIC_METADATA_DIR,
        cache_dir=TRAFFIC_SEARCH_CACHE_DIR,
        search_dims=TRAFFIC_SEARCH_DIMS,
    )
    video_url_cache.update(traffic_search_index.video_url_by_id)
    print(
        f"Caption collection index sẵn sàng: {traffic_search_index.size:,} frames / "
        f"{traffic_search_index.video_count} videos / "
        f"{len(traffic_search_index.shard_names)} embedding shards."
    )
except Exception as exc:
    traffic_search_reason = f"{type(exc).__name__}: {exc}"
    print(f"CẢNH BÁO: Traffic Search bị tắt: {traffic_search_reason}")

portable_image_index = None
portable_image_reason = None
try:
    portable_image_index = PortableJinaImageIndex(JINA_PORTABLE_IMAGE_DIR)
    video_url_cache.update(portable_image_index.video_url_by_id)
    print(
        f"Portable Jina image index sẵn sàng: {portable_image_index.ntotal:,} frames / "
        f"{portable_image_index.video_count} videos / "
        f"{len(portable_image_index.package_names)} packages."
    )
except Exception as exc:
    portable_image_reason = f"{type(exc).__name__}: {exc}"
    print(f"CẢNH BÁO: Portable Jina image search bị tắt: {portable_image_reason}")


def build_playback_info(video_id, pts_time=0):
    """Prefer a local video endpoint and fall back to the external watch URL."""
    normalized_video_id = str(video_id or "").upper()
    try:
        playback_start = max(0.0, float(pts_time or 0))
    except (TypeError, ValueError):
        playback_start = 0.0

    local_id = next(
        (alias for alias in video_id_aliases(normalized_video_id) if alias in local_video_index),
        None,
    )
    if local_id is not None:
        return {
            "playback_url": f"/videos/{local_id}",
            "playback_type": "local",
            "playback_start": playback_start,
        }

    watch_url = next(
        (
            video_url_cache.get(alias)
            for alias in video_id_aliases(normalized_video_id)
            if video_url_cache.get(alias)
        ),
        None,
    )
    if watch_url:
        separator = "&" if "?" in watch_url else "?"
        return {
            "playback_url": f"{watch_url}{separator}t={int(playback_start)}s",
            "playback_type": "youtube",
            "playback_start": playback_start,
        }

    return {
        "playback_url": None,
        "playback_type": None,
        "playback_start": playback_start,
    }

# File filtered đã nhúng OCR text. Với metadata legacy, overlay JSONL vẫn được
# hỗ trợ để tái tạo đúng cùng kết quả mà không sửa nguồn canonical.
canonical_metadata_root = (ARTIFACTS_DIR / "metadata").resolve()
ocr_text_is_embedded = (
    OCR_METADATA_PATH.name.lower()
    in {"metadata_ocr_filtered", "metadata_ocr_filtered.zip"}
    or OCR_METADATA_PATH.resolve().is_relative_to(canonical_metadata_root)
    or OCR_METADATA_PATH.resolve().is_relative_to((BASE_DIR / "metadata").resolve())
)
if OCR_TEXT_DIR.is_dir() and not ocr_text_is_embedded:
    ocr_overlay_stats = overlay_ocr_jsonl(OCR_TEXT_DIR, image_records)
    print(
        "Loaded OCR text from "
        f"{OCR_TEXT_DIR}: {ocr_overlay_stats['rows']:,} rows / "
        f"{ocr_overlay_stats['files']} shards "
        f"({ocr_overlay_stats['blank_texts']:,} blank; "
        f"filtered {ocr_overlay_stats['filtered_ticker_lines']:,} ticker lines in L21/L22)."
    )
else:
    if ocr_text_is_embedded:
        print(f"OCR text đã được nhúng và lọc trong {OCR_METADATA_PATH}.")
    else:
        print(
            f"CẢNH BÁO: Không tìm thấy OCR JSONL {OCR_TEXT_DIR}; "
            "OCR mode dùng ocr_text trong metadata canonical."
        )
ocr_data = image_records

# Hai index Jina dùng exact search trên NPY mmap và chung một thứ tự metadata.
jina_image_shard_filename = (
    "image_embeddings.npy"
    if any(JINA_VECTORS_DIR.glob("L*/image_embeddings.npy")) else None
)
jina_semantic_index = ShardedNpyIndex(
    "Jina",
    JINA_VECTORS_DIR,
    image_records,
    expected_dimension=1024,
    shard_filename=jina_image_shard_filename,
)
jina_caption_index = None
jina_caption_index_reason = "Caption embeddings chưa được tạo."
try:
    jina_caption_shard_filename = (
        "caption_embeddings.npy"
        if any(JINA_CAPTION_VECTORS_DIR.glob("L*/caption_embeddings.npy")) else None
    )
    expected_caption_shards = {f"L{number}" for number in range(21, 31)}
    if jina_caption_shard_filename:
        present_caption_shards = {
            path.parent.name
            for path in JINA_CAPTION_VECTORS_DIR.glob("L*/caption_embeddings.npy")
        }
    else:
        present_caption_shards = {
            path.stem for path in JINA_CAPTION_VECTORS_DIR.glob("L*.npy")
        }
    missing_caption_shards = sorted(expected_caption_shards - present_caption_shards)
    if missing_caption_shards:
        raise FileNotFoundError(
            "Caption embeddings chưa đầy đủ; còn thiếu "
            + ", ".join(missing_caption_shards)
            + ". Chạy embedding/jina/encode_captions.py để tiếp tục."
        )
    jina_caption_index = ShardedNpyIndex(
        "Jina Caption",
        JINA_CAPTION_VECTORS_DIR,
        image_records,
        expected_dimension=1024,
        shard_filename=jina_caption_shard_filename,
    )
    jina_caption_index_reason = (
        f"Đã map {jina_caption_index.ntotal:,} caption vectors."
    )
except (FileNotFoundError, ValueError) as exc:
    # Jina image vẫn dùng độc lập được nếu caption artifact chưa đầy đủ.
    jina_caption_index_reason = str(exc)
    print(f"Caption search chưa sẵn sàng: {exc}")
jina_text_encoder = JinaTextEncoder(device=device)
print(f"Đã map Jina image: {jina_semantic_index.ntotal:,} vector, 1024 chiều.")

print("Loading ASR metadata...")
asr_data, _, asr_video_map = load_asr_metadata(ASR_METADATA_DIR)
print(f"Loaded {len(asr_data)} ASR segments from {len(asr_video_map)} videos.")

# --- 6. XÂY DỰNG CÁC INDEX TÌM KIẾM ---

# HÀM LÀM SẠCH OCR
def clean_ocr_text(text):
    text_lower = text.lower()
    patterns = [
        r'\d{1,2}:\d{2}(:\d{2})?', 
        r'\b(htv|htvt)\d?\b',   
        r'\b(website|fanpage|youtube|tintuc|www|fb\.com)\b',
        r'\.(com|vn)',           
        r'gưu họ c',           
        r'\bfiv\b'            
    ]
    for pattern in patterns:
        text_lower = re.sub(pattern, ' ', text_lower, flags=re.IGNORECASE)
    text_lower = re.sub(r'\b[a-zA-Z]\b', ' ', text_lower)
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    return text_lower


def tokenize_ocr_text(text):
    """Tokenize OCR consistently and detach punctuation from Vietnamese words."""
    return re.findall(r"[^\W_]+", clean_ocr_text(text), flags=re.UNICODE)


def tokenize_asr_text(text):
    """Tokenize ASR without applying OCR-specific logo/time cleanup rules."""
    return re.findall(r"[^\W_]+", str(text or "").lower(), flags=re.UNICODE)
# --- KẾT THÚC HÀM ---


# Index 2: BM25 cho OCR
print("Đang xây dựng index tìm kiếm với BM25 (cho OCR)...")
bm25_ocr_index = None
if ocr_data:
    # OCR của slide/bài giảng thường dài hơn caption/logo rất nhiều. b thấp
    # giúp BM25 không phạt độ dài quá tay; coverage/phrase bonus ở
    # ocr_candidates() đảm bảo khớp đủ cụm từ vẫn đứng trên khớp một từ ngắn.
    ocr_sources = [OCR_METADATA_PATH]
    if OCR_TEXT_DIR.is_dir() and not ocr_text_is_embedded:
        ocr_sources.append(OCR_TEXT_DIR)
    bm25_ocr_index = PersistentInvertedBM25.load_or_build(
        SEARCH_INDEX_CACHE_DIR,
        "ocr",
        len(ocr_data),
        lambda index: tokenize_ocr_text(ocr_data[index].get("ocr_text", "")),
        {
            "artifacts": fingerprint_paths(ocr_sources),
            "tokenizer": "clean-ocr-v1+unicode-word-v1",
        },
        k1=1.5,
        b=0.20,
    )
    print(f"BM25 inverted index (OCR) sẵn sàng cho {len(ocr_data)} văn bản.")
else:
    print("Không có dữ liệu OCR để xây dựng index BM25.")


# Index 3: BM25 cho ASR
print("Đang xây dựng index tìm kiếm với BM25 (cho ASR)...")
bm25_asr_index = None
if asr_data:
    # Dùng cùng tokenizer/ranking policy mới của OCR nhưng không chạy các regex
    # cleanup riêng cho logo, timestamp và website của OCR.
    bm25_asr_index = PersistentInvertedBM25.load_or_build(
        SEARCH_INDEX_CACHE_DIR,
        "asr",
        len(asr_data),
        lambda index: tokenize_asr_text(asr_data[index].get("text", "")),
        {
            "artifacts": fingerprint_paths([ASR_METADATA_DIR]),
            "tokenizer": "lowercase+unicode-word-v1",
        },
        k1=1.5,
        b=0.20,
    )
    print(f"BM25 inverted index (ASR) sẵn sàng cho {len(asr_data)} văn bản.")
else:
    print("Không có dữ liệu ASR để xây dựng index BM25.")


# --- 7. TẠO FLASK APP ---
app = Flask(__name__)
CORS(app, allow_headers="*")

# HÀM HELPER ĐỂ TÌM KEYFRAME GẦN NHẤT
def find_closest_keyframe(video_id, target_time):
    if video_id not in keyframe_time_cache:
        return {"frame_n": None, "frame_idx": None}
    cache_entry = keyframe_time_cache[video_id]
    times = cache_entry["times"]
    data = cache_entry["data"]
    if not times:
        return {"frame_n": None, "frame_idx": None}
    index = bisect.bisect_left(times, target_time)
    if index == 0:
        best_match_data = data[0]
    elif index == len(times):
        best_match_data = data[-1]
    else:
        time_before = times[index - 1]
        time_after = times[index]
        if (target_time - time_before) < (time_after - target_time):
            best_match_data = data[index - 1]
        else:
            best_match_data = data[index]
    return {
        # best_match_data[0] là 'n' (tên frame), [1] là 'frame_idx'
        "frame_n": best_match_data[0], 
        "frame_idx": best_match_data[1]
    }
# --- KẾT THÚC HÀM HELPER ---

# --- 8. CÁC API ---

# (XÓA BỎ) Hàm helper get_request_data()
# def get_request_data(): ...

# (CẬP NHẬT) Hàm chuẩn hóa đường dẫn web
def get_web_path(original_path):
    return parse_keyframe_path(original_path)


def get_frame_web_path(video_id, frame_id):
    meta = metadata_cache.get(video_id, {}).get(int(frame_id), {})
    return meta.get('path')


def get_neighbor_frame_ids(video_id, frame_id, radius):
    frames = video_frame_ids.get(video_id, [])
    try:
        position = frames.index(int(frame_id))
    except ValueError:
        return []
    start = max(0, position - radius)
    end = min(len(frames), position + radius + 1)
    return frames[start:end]


# === (THÊM MỚI) QUERY EXPANSION (Groq) - Theo "[AIC2026] - Query expansion.docx", PLAN A ===
QUERY_EXPANSION_PROMPT_TEMPLATE = """Bạn là chuyên gia viết truy vấn cho mô hình Jina đa phương thức trong bài toán Video Information Retrieval.

Nhiệm vụ: Chuyển câu truy vấn tiếng Việt thành 3 biến thể tiếng Việt giàu chi tiết thị giác để đối chiếu với cả ảnh và caption tiếng Anh bằng Jina.

QUY TẮC BẮT BỘC:
1. LOẠI BỎ TỪ KHÔNG CÓ HÌNH ẢNH: Bỏ các từ chỉ cảm xúc ("vui vẻ", "thanh mát"), từ chỉ nhiệm vụ ("làm nhiệm vụ", "nghiên cứu"), địa danh chung chung ("miền Nam", "miền Tây").
2. THỊ GIÁC HÓA (Visual Concretization): Dịch các khái niệm thành mô tả hình ảnh trực quan (Ví dụ: "loài chim ở Nam Bộ" -> dịch chi tiết đặc điểm màu lông, màu mắt của chim được mô tả trong câu).
3. BẢO TOÀN THỰC THỂ (Entity Recall): KHÔNG ĐƯỢC BỎ SÓT bất kỳ đối tượng, màu sắc, trang phục, đồ vật phụ nào (như "khăn rằn", "ghe xanh", "hoa pansy", "chiếc túi giấy", "hộp đổ bóng").
4. GIỮ NGUYÊN Ý NGHĨA: KHÔNG BỊẠ THÊM các chi tiết không có trong câu gốc.

Ví dụ mẫu:
Input: "Cảnh thu hoạch dứa: một bà cụ ngồi bên giỏ dứa trò chuyện với cô gái mặc áo hồng quàng khăn rằn; xung quanh chất đầy dứa, phía sau có người phụ nữ đội nón lá cầm trái dứa và một chiếc ghe xanh đậu cạnh bờ."

JSON Output:
{{
  "dense_caption": "An elderly woman sitting next to a basket of pineapples talking to a girl wearing a pink shirt and a traditional checked scarf, surrounded by harvested pineapples, with a woman in a conical hat holding a pineapple behind them and a blue boat parked by the riverbank.",
  "structured_entities": "elderly woman, basket of pineapples, girl in pink shirt, checked scarf, woman in conical hat, blue boat, riverbank, harvested pineapples",
  "spatial_action_focus": "a girl in pink shirt and an elderly woman sitting near pineapples with a blue boat moored at the shore"
}}

Yêu cầu đầu ra cho câu truy vấn dưới đây:
- "dense_caption": Dịch toàn bộ câu sang tiếng Anh Alt-text tự nhiên, giữ lại 100% chi tiết thị giác, màu sắc, vị trí.
- "structured_entities": Liệt kê TẤT CẢ các cụm thực thể + thuộc tính (màu sắc, hình dáng) ngăn cách bằng dấu phẩy.
- "spatial_action_focus": Tóm tắt mối quan hệ không gian và hành động cốt lõi giữa các chủ thể chính.

Câu truy vấn gốc: "{query}"

Chỉ trả về 1 Object JSON duy nhất, không thêm bất kỳ dòng giải thích hay ký tự markdown nào khác:
{{"dense_caption": "...", "structured_entities": "...", "spatial_action_focus": "..."}}"""

def expand_query_with_groq(query_text):
    """Mở rộng query bảo toàn tối đa chi tiết cho Jina image/caption retrieval."""
    if not groq_client:
        return []
    try:
        prompt = QUERY_EXPANSION_PROMPT_TEMPLATE.format(query=query_text)
        raw = groq_generate(prompt).strip()
        raw = re.sub(r'^```(json)?|```$', '', raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)

        # Lấy đầy đủ 3 biến thể giàu chi tiết
        variants = [
            parsed.get('dense_caption', ''),
            parsed.get('structured_entities', ''),
            parsed.get('spatial_action_focus', '')
        ]

        # Lọc bỏ chuỗi rỗng
        variants = [v.strip() for v in variants if v and v.strip()]

        # Loại bỏ các biến thể trùng lặp nếu có
        variants = list(dict.fromkeys(variants))

        print(f"[QueryExpansion] Raw: '{query_text}'")
        for idx, var in enumerate(variants, 1):
            print(f"  └─ Variant {idx}: {var}")

        return variants
    except Exception as e:
        print(f"Lỗi khi mở rộng câu truy vấn bằng Groq: {e}")
        return []


# (THÊM MỚI) API /expand_query - chỉ sinh 3 biến thể để người dùng chọn, KHÔNG tự search.
# Trước đây tick checkbox là tự động search cả 3 biến thể + gộp RRF (người dùng không biết đã tìm
# bằng câu gì). Giờ tách riêng: bấm nút "Mở rộng" -> hiện 3 lựa chọn -> người dùng bấm chọn 1 cái ->
# cái đó trở thành query rồi search bình thường qua /search.
@app.route('/expand_query', methods=['POST'])
def expand_query_endpoint():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request phải là JSON"}), 400
        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({"variants": []})
        variants = expand_query_with_groq(query_text)
        return jsonify({"variants": variants})
    except Exception as e:
        print(f"Lỗi trong /expand_query: {e}")
        return jsonify({"error": str(e)}), 500

def reciprocal_rank_fusion(ranked_id_lists, k=60, weights=None):
    """RRF: mỗi list là danh sách index đã sort tốt nhất trước. Trả (idx, fused_score) sort giảm dần.
    weights (tuỳ chọn): trọng số tương ứng từng list theo thứ tự trong ranked_id_lists, mặc định bằng nhau
    (dùng cho Fusion search: tỷ lệ Jina Hybrid/OCR/ASR do người dùng chỉnh)."""
    if weights is None:
        weights = [1.0] * len(ranked_id_lists)
    scores = {}
    for w, ranked_ids in zip(weights, ranked_id_lists):
        for rank, idx in enumerate(ranked_ids):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_result_rankings(ranked_results, top_k, rrf_k=60.0):
    """Fuse API result dictionaries by normalized video/frame identity."""
    scores = {}
    payloads = {}
    sources = collections.defaultdict(set)
    for results in ranked_results:
        seen = set()
        for rank, item in enumerate(results):
            frame_idx = item.get("frame_idx")
            if frame_idx is None:
                path_stem = Path(str(item.get("path") or "")).stem
                if not path_stem.isdigit():
                    continue
                frame_idx = int(path_stem)
            key = normalized_frame_key(item.get("videoId", ""), int(frame_idx))
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (float(rrf_k) + rank + 1.0)
            if key not in payloads:
                payloads[key] = dict(item)
                payloads[key]["source_score"] = float(item.get("score", 0.0))
            else:
                for field in ("caption", "vehicle_count", "vehicle_counts", "detection_available"):
                    if field in item and field not in payloads[key]:
                        payloads[key][field] = item[field]
            sources[key].update(item.get("matched_by") or [])

    ranked = sorted(scores, key=lambda key: scores[key], reverse=True)[:max(1, int(top_k))]
    output = []
    for key in ranked:
        item = payloads[key]
        item["score"] = float(scores[key])
        item["matched_by"] = sorted(sources[key])
        output.append(item)
    return output


def enrich_portable_results(results):
    """Overlay exact timestamps/captions from shared M/N metadata when present."""
    if traffic_search_index is not None:
        for item in results:
            asset = traffic_search_index.image_asset_for_video_frame(
                item["videoId"], item["frame_idx"]
            )
            item["image_available"] = asset is not None
            enrichment = traffic_search_index.metadata_for_video_frame(
                item["videoId"], item["frame_idx"]
            )
            if enrichment is not None:
                item["pts_time"] = float(enrichment["pts_time"])
                if enrichment.get("caption"):
                    item["caption"] = enrichment["caption"]
    return results


def search_supplemental_semantic(query_text, semantic_model, query_vector, top_k):
    """Search M/N/S image packages and optionally fuse the M/N caption rank."""
    image_results = enrich_portable_results(
        portable_image_index.search(query_vector, top_k=top_k)
        if portable_image_index is not None else []
    )
    caption_results = []
    if traffic_search_index is not None and semantic_model == "jina-hybrid":
        caption_results = traffic_search_index.search(
            query_text,
            query_vector,
            top_k=min(traffic_search_index.size, top_k),
        )
        for item in caption_results:
            item["batch"] = "batch2"

    if image_results and caption_results:
        return fuse_result_rankings([image_results, caption_results], top_k)
    if image_results:
        return image_results[:top_k]
    if caption_results:
        return caption_results[:top_k]
    # Compatibility fallback while a machine has old captions but has not yet
    # downloaded the portable image packages.
    if traffic_search_index is not None:
        fallback = traffic_search_index.search(
            query_text, query_vector, top_k=min(traffic_search_index.size, top_k)
        )
        for item in fallback:
            item["batch"] = "batch2"
        return fallback
    return []
# === (KẾT THÚC) QUERY EXPANSION ===

# === (CẬP NHẬT) OCR/ASR: bỏ hẳn Elasticsearch, dùng thẳng BM25 tự viết (đã build sẵn lúc khởi động) ===
def coverage_phrase_bm25_scores(
    bm25_index, tokens_for_document, tokenized_query, search_size
):
    """BM25 scores with length-independent term coverage and exact phrase bonus."""
    tokenized_query = list(dict.fromkeys(tokenized_query))
    if not tokenized_query:
        return None

    scores, match_counts = bm25_index.get_scores_with_match_counts(tokenized_query)
    known_query_terms = [
        term for term in tokenized_query if bm25_index.has_term(term)
    ]
    if not known_query_terms:
        return None

    query_weight = sum(bm25_index.idf_for(term) for term in known_query_terms)
    coverage = match_counts.astype(np.float64) / len(known_query_terms)
    scores += query_weight * np.square(coverage)

    # Chỉ dò vị trí phrase trong một pool rộng sau coverage để request vẫn nhanh.
    if len(tokenized_query) > 1:
        rerank_size = min(
            bm25_index.doc_count,
            max(int(search_size) * 20, 2000),
        )
        if rerank_size < bm25_index.doc_count:
            rerank_indices = np.argpartition(scores, -rerank_size)[-rerank_size:]
        else:
            rerank_indices = np.arange(bm25_index.doc_count)
        phrase_length = len(tokenized_query)
        phrase_bonus = query_weight * 1.5
        for index in rerank_indices:
            document = tokens_for_document(int(index))
            if any(
                document[start:start + phrase_length] == tokenized_query
                for start in range(len(document) - phrase_length + 1)
            ):
                scores[int(index)] += phrase_bonus
    return scores


def ocr_candidates(query_text, search_size):
    """OCR ranking: BM25 nhẹ length penalty + query coverage + phrase bonus."""
    if not bm25_ocr_index:
        return []
    scores = coverage_phrase_bm25_scores(
        bm25_ocr_index,
        lambda index: tokenize_ocr_text(ocr_data[index].get("ocr_text", "")),
        tokenize_ocr_text(query_text),
        search_size,
    )
    if scores is None:
        return []

    top_k_indices = np.argsort(scores)[::-1][:search_size]
    out = []
    for i in top_k_indices:
        score = scores[i]
        if score <= 0:
            continue
        path = ocr_data[i].get('path', '')
        if not path:
            continue
        out.append((float(score), path))
    return out


def asr_candidates(query_text, search_size):
    """ASR ranking dùng cùng coverage/phrase policy với OCR."""
    if not bm25_asr_index:
        return []
    scores = coverage_phrase_bm25_scores(
        bm25_asr_index,
        lambda index: tokenize_asr_text(asr_data[index].get("text", "")),
        tokenize_asr_text(query_text),
        search_size,
    )
    if scores is None:
        return []
    top_k_indices = np.argsort(scores)[::-1][:search_size]
    out = []
    for i in top_k_indices:
        score = scores[i]
        if score <= 0:
            continue
        doc = asr_data[i]
        out.append({
            "video_id": doc["video_id"],
            "text": doc["text"],
            "start": doc["start"],
            "end": doc["end"],
            "score": float(score),
        })
    return out
# === (KẾT THÚC) OCR/ASR SEARCH HELPERS ===


SEMANTIC_MODEL_LABELS = {
    "jina": "Jina Embeddings v5 · Tất cả collection",
    "jina-hybrid": "Jina Hybrid · Tất cả collection (RRF)",
}


def _encode_semantic_query_uncached(query_text, semantic_model):
    if semantic_model in {"jina", "jina-hybrid"}:
        return jina_text_encoder.encode(query_text)
    raise ValueError(
        f"semantic_model không hợp lệ: {semantic_model!r}. "
        f"Chọn một trong {sorted(SEMANTIC_MODEL_LABELS)}."
    )


@lru_cache(maxsize=SEMANTIC_QUERY_CACHE_SIZE)
def _cached_semantic_query_vector(query_text, semantic_model):
    vector = np.asarray(
        _encode_semantic_query_uncached(query_text, semantic_model),
        dtype=np.float32,
    ).copy()
    vector.setflags(write=False)
    return vector


def encode_semantic_query(query_text, semantic_model):
    """Encode a text query with a bounded in-process LRU cache."""
    # Return a copy so downstream code cannot corrupt the cached vector.
    return _cached_semantic_query_vector(query_text, semantic_model).copy()


def search_semantic_vectors(semantic_model, query_vector, top_k):
    if semantic_model == "jina":
        return jina_semantic_index.search(query_vector, top_k)
    if semantic_model == "jina-hybrid":
        if jina_caption_index is None:
            raise ModelUnavailableError(jina_caption_index_reason)

        # Hai nhánh có phân phối cosine khác nhau (image vs caption), nên gộp
        # thứ hạng bằng RRF thay vì cộng trực tiếp raw similarity score.
        # Normal search benefits from a wider pool; TRAKE already asks for
        # 10k candidates, so cap here instead of expanding to 50k per branch.
        branch_k = max(min(int(top_k) * 5, 10000), 100)
        _, image_indices = jina_semantic_index.search(query_vector, branch_k)
        _, caption_indices = jina_caption_index.search(query_vector, branch_k)
        image_rank = [int(index_id) for index_id in image_indices[0] if index_id >= 0]
        caption_rank = [
            int(index_id) for index_id in caption_indices[0] if index_id >= 0
        ]
        fused = reciprocal_rank_fusion(
            [image_rank, caption_rank],
            weights=[1.0, 1.0],
        )[: max(1, int(top_k))]
        return (
            np.asarray([[score for _, score in fused]], dtype=np.float32),
            np.asarray([[index_id for index_id, _ in fused]], dtype=np.int64),
        )
    raise ValueError(f"semantic_model không hợp lệ: {semantic_model!r}")


@lru_cache(maxsize=SEMANTIC_RESULT_CACHE_SIZE)
def _cached_semantic_text_search(query_text, semantic_model, top_k):
    vector = _cached_semantic_query_vector(query_text, semantic_model)
    distances, indices = search_semantic_vectors(
        semantic_model, vector, int(top_k)
    )
    distances = np.asarray(distances).copy()
    indices = np.asarray(indices).copy()
    distances.setflags(write=False)
    indices.setflags(write=False)
    return distances, indices


def search_semantic_text(query_text, semantic_model, top_k):
    """Encode and search while caching repeated query/model/top-k requests."""
    distances, indices = _cached_semantic_text_search(
        str(query_text), str(semantic_model), int(top_k)
    )
    return distances.copy(), indices.copy()


@app.route('/semantic_models', methods=['GET'])
def semantic_models_status():
    jina_available, jina_reason = jina_text_encoder.availability()
    caption_available = jina_available and jina_caption_index is not None
    caption_reason = (
        jina_caption_index_reason if jina_available else jina_reason
    )
    return jsonify({
        "models": {
            "jina": {
                "label": SEMANTIC_MODEL_LABELS["jina"],
                "available": jina_available,
                "dimension": jina_semantic_index.d,
                "vectors": jina_semantic_index.ntotal + (
                    portable_image_index.ntotal
                    if portable_image_index is not None else 0
                ),
                "reason": portable_image_reason or jina_reason,
            },
            "jina-hybrid": {
                "label": SEMANTIC_MODEL_LABELS["jina-hybrid"],
                "available": caption_available,
                "dimension": 1024,
                "vectors": (
                    (jina_caption_index.ntotal if jina_caption_index is not None else 0)
                    + (traffic_search_index.size if traffic_search_index is not None else 0)
                    + (portable_image_index.ntotal if portable_image_index is not None else 0)
                ),
                "reason": caption_reason,
            },
        }
    })

# API /search
# (TRONG app.py)
# API /search (ĐÃ CẬP NHẬT)
@app.route('/search', methods=['POST'])
def search():
    try:
        # 1. Lấy dữ liệu request
        data = request.get_json()
        if data is None:
             return jsonify({"error": "Request phải là JSON"}), 400
             
        query_text = str(data.get('query', '')).strip()
        if not query_text:
            return jsonify({"error": "Query không được để trống."}), 400
        top_k = int(data.get('top_k', 50))
        semantic_model = str(data.get('semantic_model', 'jina')).strip().lower()
        if semantic_model not in SEMANTIC_MODEL_LABELS:
            return jsonify({
                "error": f"semantic_model không hợp lệ: {semantic_model!r}",
                "allowed_models": sorted(SEMANTIC_MODEL_LABELS),
            }), 400
        # (SỬA LỖI) Xử lý 'group' (là boolean true/false)
        group_results = data.get('group', False) 
        
        # === (LOGIC MỚI) KIỂM TRA XEM QUERY CÓ PHẢI LÀ VIDEO ID KHÔNG ===
        
        # Chuẩn hóa query (ví dụ: " l22_v002 " -> "L22_V002")
        video_id_query = query_text.strip().upper() 
        
        # Kiểm tra xem query này có nằm trong danh sách video ID ta có không
        # Video IDs are discovered directly from OCR metadata.
        if video_id_query in metadata_cache:
            print(f"Video ID search detected: {video_id_query}")
            results = []
            summary = {}
            video_meta = metadata_cache.get(video_id_query, {})

            for frame_id, meta in video_meta.items():
                web_path = meta.get('path')
                if not web_path:
                    continue
                pts_time = float(meta.get('pts_time', 0) or 0)
                results.append({
                    "path": web_path,
                    "videoId": video_id_query,
                    "score": pts_time,
                    "pts_time": pts_time
                })

            summary[video_id_query] = len(results)
            final_results = sorted(results, key=lambda item: item['pts_time'])[:top_k]
            if group_results:
                return jsonify({
                    "results": {video_id_query: final_results},
                    "summary": summary
                })
            return jsonify({"results": final_results, "summary": summary})

        if (
            portable_image_index is not None
            and portable_image_index.keyframe_map(video_id_query) is not None
        ):
            results = enrich_portable_results(
                portable_image_index.results_for_video(video_id_query, top_k)
            )
            summary = {results[0]["videoId"]: len(results)} if results else {}
            if group_results and results:
                return jsonify({"results": {results[0]["videoId"]: results}, "summary": summary})
            return jsonify({"results": results, "summary": summary})

        if (
            traffic_search_index is not None
            and video_id_query in traffic_search_index.rows_by_video
        ):
            rows = traffic_search_index.rows_by_video[video_id_query]
            results = [{
                "path": traffic_search_index.web_paths[row],
                "videoId": video_id_query,
                "score": float(traffic_search_index.timestamps[row]),
                "pts_time": float(traffic_search_index.timestamps[row]),
                "frame_idx": int(traffic_search_index.frame_indices[row]),
                "batch": "batch2",
            } for row in rows[:top_k]]
            summary = {video_id_query: len(results)}
            if group_results:
                return jsonify({"results": {video_id_query: results}, "summary": summary})
            return jsonify({"results": results, "summary": summary})

        # === (KẾT THÚC LOGIC MỚI) ===
        
        # Nếu không phải là Video ID, chạy logic tìm kiếm semantic CŨ
        print(
            f"Đang tìm kiếm semantic bằng {SEMANTIC_MODEL_LABELS[semantic_model]} "
            f"cho: '{query_text}'"
        )

        search_query = query_text
        query_translated = False
        translation_reason = ""

        pool_k = top_k * 5 if group_results else top_k

        semantic_score_by_idx = {}
        distances, indices = search_semantic_text(
            search_query, semantic_model, pool_k
        )
        ordered_indices = [int(i) for i in indices[0] if int(i) >= 0]
        for i, dist in zip(indices[0], distances[0]):
            if int(i) >= 0:
                semantic_score_by_idx[int(i)] = float(dist)

        batch1_results = []

        for i in ordered_indices:
            original_path = image_records[int(i)]['path']
            web_path, video_id, frame_n_str = get_web_path(original_path)

            # (SỬA LỖI) Thêm check frame_n_str (không phải None)
            if web_path and frame_n_str:
                frame_n_int = int(frame_n_str)
                meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0

                batch1_results.append({
                    "path": web_path,
                    "videoId": video_id,
                    "score": semantic_score_by_idx.get(i, 0.0),
                    "pts_time": float(pts_time),
                    "batch": "batch1",
                })

        batch2_results = []
        if semantic_model in {"jina", "jina-hybrid"}:
            query_vector = _cached_semantic_query_vector(search_query, semantic_model)
            batch2_results = search_supplemental_semantic(
                query_text,
                semantic_model,
                query_vector,
                top_k=pool_k,
            )

        # Scores của hai kho không cùng phân phối (Batch 1 có thể là cosine
        # hoặc Hybrid RRF; Batch 2 dùng caption semantic + detection). Trộn
        # theo rank giúp hai batch cùng có cơ hội xuất hiện mà không cần giả
        # định hai raw score có cùng thang đo.
        if batch2_results:
            results = merge_ranked_batches(batch1_results, batch2_results)
        else:
            results = batch1_results

        summary = {}
        for item in results:
            video_id = item["videoId"]
            summary[video_id] = summary.get(video_id, 0) + 1

        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in results:
                video_id = res['videoId']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)
            
            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                sorted_items = sorted(items, key=lambda x: x['pts_time'])
                final_grouped_results[video_id] = sorted_items[:top_k] 
            
            return jsonify({
                "results": final_grouped_results,
                "summary": sorted_summary,
                "semantic_model": semantic_model,
                "search_query": search_query,
                "query_translated": query_translated,
                "translation_reason": translation_reason,
                "searched_batches": ["batch1", "batch2", "final"] if batch2_results else ["batch1"],
            })
        else:
            final_results = results[:top_k]
            return jsonify({
                "results": final_results,
                "summary": sorted_summary,
                "semantic_model": semantic_model,
                "search_query": search_query,
                "query_translated": query_translated,
                "translation_reason": translation_reason,
                "searched_batches": ["batch1", "batch2", "final"] if batch2_results else ["batch1"],
            })

    except ModelUnavailableError as e:
        print(f"Model semantic chưa sẵn sàng: {e}")
        return jsonify({"error": str(e), "semantic_model": semantic_model}), 503
    except ValueError as e:
        print(f"Request /search không hợp lệ: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Lỗi trong /search: {e}")
        return jsonify({"error": str(e)}), 500

# (CẬP NHẬT) API /search_similar_image - Tích hợp YOLOv8n Auto-Crop
@app.route('/search_similar_image', methods=['POST'])
def search_similar_image():
    try:
        # Lấy FormData trực tiếp
        data = request.form.to_dict()
        top_k = int(data.get('top_k', 50))
        group_results = data.get('group', 'false').lower() == 'true'
        # (THÊM MỚI) Tham số auto_crop từ nút Toggle trên giao diện
        auto_crop = data.get('auto_crop', 'false').lower() == 'true'
        if 'image_file' not in request.files:
            return jsonify({"error": "Không có tệp ảnh nào được tải lên."}), 400

        file = request.files['image_file']
        original_image = Image.open(io.BytesIO(file.read())).convert("RGB")
        target_image = original_image  # Mặc định dùng ảnh gốc

        # (THÊM MỚI) --- BƯỚC TIỀN XỬ LÝ: YOLO AUTO-CROP ---
        if auto_crop and yolo_model is not None:
            print("[AutoCrop] Đang detect vật thể bằng YOLOv8n...")
            with torch.no_grad():
                yolo_results = yolo_model(original_image, verbose=False)

            boxes = yolo_results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # --- Chiến lược chọn box: ưu tiên diện tích lớn gần trung tâm ---
                img_w, img_h = original_image.size
                img_cx, img_cy = img_w / 2.0, img_h / 2.0

                best_box = None
                best_score = -1.0

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    area = (x2 - x1) * (y2 - y1)
                    box_cx = (x1 + x2) / 2.0
                    box_cy = (y1 + y2) / 2.0

                    # Khoảng cách từ trung tâm box đến trung tâm ảnh (chuẩn hóa)
                    dist_to_center = ((box_cx - img_cx) ** 2 + (box_cy - img_cy) ** 2) ** 0.5
                    max_dist = ((img_w / 2) ** 2 + (img_h / 2) ** 2) ** 0.5
                    center_score = 1.0 - (dist_to_center / max_dist) if max_dist > 0 else 1.0

                    # Diện tích chuẩn hóa
                    area_normalized = area / (img_w * img_h) if (img_w * img_h) > 0 else 0

                    # Tổng hợp: 50% confidence + 30% diện tích + 20% gần trung tâm
                    combined_score = 0.5 * conf + 0.3 * area_normalized + 0.2 * center_score

                    if combined_score > best_score:
                        best_score = combined_score
                        best_box = box

                if best_box is not None:
                    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                    cls_id = int(best_box.cls[0])
                    conf = float(best_box.conf[0])
                    class_name = yolo_results[0].names.get(cls_id, f"class_{cls_id}")

                    # Padding 15px, clamp trong giới hạn ảnh
                    PADDING = 15
                    x1_pad = max(0, int(x1) - PADDING)
                    y1_pad = max(0, int(y1) - PADDING)
                    x2_pad = min(img_w, int(x2) + PADDING)
                    y2_pad = min(img_h, int(y2) + PADDING)

                    target_image = original_image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
                    print(f"[AutoCrop] Đã crop vật thể '{class_name}' với độ tin cậy {conf:.2f} | Box: [{x1_pad},{y1_pad},{x2_pad},{y2_pad}]")
                else:
                    print("[AutoCrop] Không tìm được box tốt nhất. Dùng ảnh gốc.")
            else:
                print("[AutoCrop] YOLO không phát hiện vật thể nào. Fallback về ảnh gốc.")

            # Giải phóng VRAM sau YOLO inference
            del yolo_results
            if device == "cuda":
                torch.cuda.empty_cache()
        # --- KẾT THÚC BƯỚC TIỀN XỬ LÝ ---

        # Ảnh query và toàn bộ keyframe đều dùng cùng Jina retrieval space.
        query_vector = jina_text_encoder.encode_image(target_image)

        # Dọn dẹp sau Jina inference
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        pool_k = top_k * 5 if group_results else top_k
        distances, indices = jina_semantic_index.search(query_vector, pool_k)
        ordered = [(int(i), float(dist)) for i, dist in zip(indices[0], distances[0])]

        results = []
        summary = {}

        for i, score in ordered:
            original_path = image_records[int(i)]['path']
            web_path, video_id, frame_n_str = get_web_path(original_path)

            if web_path and frame_n_str:
                frame_n_int = int(frame_n_str)
                meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0

                results.append({
                    "path": web_path,
                    "videoId": video_id,
                    "score": float(score),
                    "pts_time": float(pts_time),
                    "batch": "batch1",
                })
                summary[video_id] = summary.get(video_id, 0) + 1

        if portable_image_index is not None:
            portable_results = enrich_portable_results(
                portable_image_index.search(query_vector, top_k=pool_k)
            )
            results.extend(portable_results)
            for item in portable_results:
                video_id = item["videoId"]
                summary[video_id] = summary.get(video_id, 0) + 1

        # Cùng model/revision, cùng document-side image space nên cosine của
        # Batch 1, Batch 2 và S01 có thể so sánh trực tiếp.
        results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in results:
                video_id = res['videoId']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)

            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                sorted_items = sorted(items, key=lambda x: x['pts_time'])
                final_grouped_results[video_id] = sorted_items[:top_k]

            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            final_results = results[:top_k]
            return jsonify({"results": final_results, "summary": sorted_summary})

    except ModelUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"Lỗi trong /search_similar_image: {e}")
        return jsonify({"error": str(e)}), 500


# (CẬP NHẬT) API /search_ocr
@app.route('/search_ocr', methods=['POST'])
def search_ocr():
    try:
        # (SỬA LỖI) Lấy JSON trực tiếp
        data = request.get_json()
        if data is None:
             return jsonify({"error": "Request phải là JSON"}), 400
             
        query_text = data['query'].lower()
        top_k = int(data.get('top_k', 50))
        # (SỬA LỖI) Xử lý 'group' (là boolean true/false)
        group_results = data.get('group', False)

        if not query_text:
            return jsonify({"results": [], "summary": {}})

        search_size = top_k * 5 if group_results else top_k

        candidates = ocr_candidates(query_text, search_size)
        if not candidates and not bm25_ocr_index:
            return jsonify({"error": "Chưa có dữ liệu OCR (bm25_ocr_index chưa khởi tạo)."}), 500

        results = []
        summary = {}

        for score, original_path in candidates:
            # Chạy tiếp luồng xử lý Web Path và Metadata của ông
            web_path, video_id, frame_n_str = get_web_path(original_path)

            if web_path and frame_n_str:
                frame_n_int = int(frame_n_str)
                meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0

                results.append({
                    "path": web_path,
                    "videoId": video_id,
                    "score": float(score),
                    "pts_time": float(pts_time)
                })
                summary[video_id] = summary.get(video_id, 0) + 1
        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in results:
                video_id = res['videoId']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)
            
            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                sorted_items = sorted(items, key=lambda x: x['pts_time'])
                final_grouped_results[video_id] = sorted_items[:top_k]
            
            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            final_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
            return jsonify({"results": final_results, "summary": sorted_summary})
            
    except Exception as e: 
        print(f"Lỗi trong /search_ocr: {e}")
        return jsonify({"error": str(e)}), 500

# === (VIẾT LẠI) TRAKE - TEMPORAL ALIGNMENT ===
# Bản cũ tìm frame mà TẤT CẢ các phần mô tả cùng xuất hiện trong một cửa sổ hẹp (±window_size frame).
# Điều đó sai bản chất TRAKE: các sự kiện trong một chuỗi (chạy đà -> giậm nhảy -> bay qua xà ->
# tiếp đất) nằm RẢI RÁC theo thời gian, có thể cách nhau hàng trăm frame, nên gần như không bao giờ
# "giao nhau" trong một cửa sổ hẹp.
#
# Bản mới làm đúng 2 giai đoạn theo mô tả của BTC:
#   1. Retrieval  - tìm video chứa toàn bộ chuỗi sự kiện (gộp điểm mọi sự kiện theo từng video).
#   2. Alignment  - trong mỗi video, chọn cho mỗi sự kiện đúng 1 keyframe sao cho thứ tự thời gian
#                   được giữ nguyên và tổng điểm Jina Hybrid lớn nhất,
#                   ĐỒNG THỜI phạt nặng khi 2 sự kiện liên tiếp cách nhau quá xa về thời gian.
# Giai đoạn 2 là bài toán quy hoạch động (DP) - xem _align_event_sequence().
TRAKE_DEFAULT_MAX_GAP_SECONDS = 30.0   # trong khoảng này thì không phạt
TRAKE_DEFAULT_GAP_PENALTY = 0.01       # phạt mỗi giây vượt ngưỡng; vượt 30s ~ mất trọn 1 sự kiện khớp


def _align_event_sequence(candidate_frames, frame_times, event_scores, n_events,
                          max_gap_seconds=TRAKE_DEFAULT_MAX_GAP_SECONDS,
                          gap_penalty_per_sec=TRAKE_DEFAULT_GAP_PENALTY):
    """Chọn chuỗi frame tăng dần, tối đa hoá tổng điểm Jina Hybrid trừ tiền phạt
    khoảng cách thời gian giữa 2 sự kiện liên tiếp.

    Một chuỗi hành động (chạy đà -> giậm nhảy -> ...) diễn ra liên tục trong vài chục giây, nên nếu
    chỉ ràng buộc "frame sau > frame trước" thì DP hay ghép các sự kiện cách nhau vài phút - vốn là
    những cảnh không liên quan trong cùng video. Tiền phạt tuyến tính phần vượt quá max_gap_seconds
    khiến các chuỗi rời rạc như vậy tụt hạng, nhưng vẫn không loại hẳn (phòng khi đáp án thật hơi thưa).

    candidate_frames: list frame_n (int) đã sort tăng dần - các keyframe ứng viên của 1 video.
    frame_times:      list pts_time (giây) song song với candidate_frames, cũng tăng dần.
    event_scores:     dict[(event_index, frame_n)] -> điểm Jina Hybrid. Thiếu key = sự kiện đó không
                      khớp frame đó (tính 0 điểm, vẫn cho chọn để giữ chuỗi liền mạch).
    Trả (total_score, [frame_n cho từng sự kiện]) hoặc None nếu không xếp được chuỗi hợp lệ.
    """
    m = len(candidate_frames)
    if m < n_events:
        return None  # không đủ frame để xếp N mốc khác nhau theo thứ tự tăng dần

    NEG = -1e18  # dùng số hữu hạn thay -inf để tránh NaN khi trừ tiền phạt
    times = np.asarray(frame_times, dtype=np.float64)

    # pen[i][k] = tiền phạt khi nhảy từ frame k sang frame i (chỉ tính phần vượt quá max_gap_seconds)
    gaps = times[:, None] - times[None, :]
    pen = np.maximum(gaps - max_gap_seconds, 0.0) * gap_penalty_per_sec
    # Chỉ cho phép k < i để giữ đúng thứ tự thời gian của chuỗi sự kiện
    allowed = np.tril(np.ones((m, m), dtype=bool), k=-1)

    dp = np.full((n_events, m), NEG, dtype=np.float64)
    parent = np.full((n_events, m), -1, dtype=np.int64)

    dp[0] = np.array([event_scores.get((0, fn), 0.0) for fn in candidate_frames], dtype=np.float64)

    for j in range(1, n_events):
        # vals[i][k] = điểm tốt nhất tới sự kiện j-1 tại frame k, trừ tiền phạt khi nhảy sang frame i
        vals = np.where(allowed, dp[j - 1][None, :] - pen, NEG)
        best_prev_i = vals.argmax(axis=1)
        best_prev = vals[np.arange(m), best_prev_i]
        own = np.array([event_scores.get((j, fn), 0.0) for fn in candidate_frames], dtype=np.float64)
        no_path = best_prev <= NEG / 2  # không có frame hợp lệ nào đứng trước
        dp[j] = np.where(no_path, NEG, best_prev + own)
        parent[j] = np.where(no_path, -1, best_prev_i)

    last = n_events - 1
    best_i = int(dp[last].argmax())
    if dp[last][best_i] <= NEG / 2:
        return None

    chosen = [None] * n_events
    i = best_i
    for j in range(last, -1, -1):
        chosen[j] = candidate_frames[i]
        i = int(parent[j][i])
        if i < 0 and j > 0:
            return None  # chuỗi truy vết bị đứt (không nên xảy ra)
    return float(dp[last][best_i]), chosen


@app.route('/search_trake_02', methods=['POST'])
def search_trake_02():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request phải là JSON"}), 400

        top_k_final = int(data.get('top_k', 50))
        # Mỗi phần tử là MỘT SỰ KIỆN, theo đúng thứ tự thời gian trong video.
        # UI gửi mảng 'events' (mỗi sự kiện một ô nhập riêng); vẫn chấp nhận chuỗi 'query'
        # ngăn bằng ';' để gọi API trực tiếp bằng script/curl cho tiện.
        raw_events = data.get('events')
        if isinstance(raw_events, list):
            parts = [str(p).strip() for p in raw_events if str(p).strip()]
        else:
            parts = [p.strip() for p in data.get('query', '').split(';') if p.strip()]

        if len(parts) < 2:
            return jsonify({"error": "TRAKE cần ít nhất 2 sự kiện "
                                     "(ví dụ: 'chạy đà', 'giậm nhảy', 'bay qua xà', 'tiếp đất')"}), 400

        n_events = len(parts)
        # Số sự kiện tối thiểu phải thực sự khớp thì video mới được giữ lại (cho phép hụt 1 sự kiện)
        min_events = int(data.get('min_events', max(2, n_events - 1)))
        # Ngưỡng khoảng cách thời gian giữa 2 sự kiện liên tiếp: trong ngưỡng thì không phạt,
        # vượt bao nhiêu giây thì trừ điểm bấy nhiêu * gap_penalty_per_sec.
        max_gap_seconds = float(data.get('max_gap_seconds', TRAKE_DEFAULT_MAX_GAP_SECONDS))
        gap_penalty_per_sec = float(data.get('gap_penalty_per_sec', TRAKE_DEFAULT_GAP_PENALTY))
        pool_k = max(top_k_final * 100, 10000)

        print(f"[TRAKE] Temporal alignment: {n_events} sự kiện, pool_k={pool_k}, "
              f"min_events={min_events}, max_gap={max_gap_seconds}s, penalty={gap_penalty_per_sec}/s")

        # video_event_scores[video_id][(event_index, frame_n)] = điểm Hybrid tốt nhất
        video_event_scores = collections.defaultdict(dict)
        # video_matched_events[video_id] = tập các event_index thực sự có hit trong video đó
        video_matched_events = collections.defaultdict(set)

        # Jina đa ngôn ngữ nhận trực tiếp toàn bộ sự kiện tiếng Việt.
        event_queries = parts
        for i, q in enumerate(event_queries):
            print(f"  [Sự kiện {i + 1}/{n_events}] '{q}'")

        query_vectors = jina_text_encoder.encode_texts(event_queries)
        for event_index in range(n_events):
            distances, indices = search_semantic_vectors(
                "jina-hybrid", query_vectors[event_index:event_index + 1], pool_k
            )
            for idx, dist in zip(indices[0], distances[0]):
                idx = int(idx)
                if idx < 0:
                    continue
                web_path, video_id, frame_n_str = get_web_path(image_records[int(idx)]['path'])
                if not web_path or not frame_n_str or video_id == "N/A":
                    continue
                frame_n = int(frame_n_str)
                key = (event_index, frame_n)
                # RRF tối đa xấp xỉ 2/61. Scale về gần [0, 1] để giữ nguyên
                # ý nghĩa của gap_penalty_per_sec trong thuật toán alignment.
                score = float(dist) * 30.5
                scores_of_video = video_event_scores[video_id]
                if score > scores_of_video.get(key, float('-inf')):
                    scores_of_video[key] = score
                video_matched_events[video_id].add(event_index)

        # Giai đoạn 2: căn chỉnh thời gian trong từng video ứng viên
        sequences = []
        for video_id, event_scores in video_event_scores.items():
            if len(video_matched_events[video_id]) < min_events:
                continue

            video_meta = metadata_cache.get(video_id, {})
            # Chỉ giữ frame có pts_time thật - không có mốc thời gian thì không tính được khoảng cách
            candidate_frames, candidate_times = [], []
            for frame_n in sorted({frame_n for _, frame_n in event_scores.keys()}):
                meta = video_meta.get(frame_n) or {}
                pts_time = meta.get('pts_time')
                if pts_time is None:
                    continue
                candidate_frames.append(frame_n)
                candidate_times.append(float(pts_time))
            if len(candidate_frames) < n_events:
                continue

            aligned = _align_event_sequence(candidate_frames, candidate_times, event_scores, n_events,
                                            max_gap_seconds=max_gap_seconds,
                                            gap_penalty_per_sec=gap_penalty_per_sec)
            if aligned is None:
                continue
            total_score, chosen_frames = aligned

            events_out = []
            matched_count = 0
            prev_time = None
            for event_index, frame_n in enumerate(chosen_frames):
                is_matched = (event_index, frame_n) in event_scores
                if is_matched:
                    matched_count += 1
                meta = video_meta.get(frame_n, {}) or {}
                pts_time = float(meta.get('pts_time', 0) or 0)
                frame_idx = meta.get('frame_idx')
                # Khoảng cách tới sự kiện liền trước -> UI cảnh báo khi vượt ngưỡng
                gap_from_prev = None if prev_time is None else round(pts_time - prev_time, 2)
                prev_time = pts_time
                events_out.append({
                    "event_index": event_index,
                    "query": parts[event_index],
                    "path": get_frame_web_path(video_id, frame_n),
                    "videoId": video_id,
                    "frame_n": int(frame_n),
                    # frame_idx = chỉ số frame thật trong video -> dùng thẳng để nộp đáp án TRAKE
                    "frame_idx": int(frame_idx) if frame_idx is not None else None,
                    "pts_time": pts_time,
                    "gap_from_prev": gap_from_prev,
                    "score": float(event_scores.get((event_index, frame_n), 0.0)),
                    "matched": is_matched
                })

            sequences.append({
                "videoId": video_id,
                "score": float(total_score),
                "matched_events": matched_count,
                "total_events": n_events,
                # Tổng thời lượng chuỗi - chuỗi càng gọn càng đáng tin cho một hành động liên tục
                "span_seconds": round(events_out[-1]["pts_time"] - events_out[0]["pts_time"], 2),
                "events": events_out
            })

        sequences.sort(key=lambda s: -s['score'])
        final_sequences = sequences[:top_k_final]
        print(f"[TRAKE] {len(sequences)} video xếp được chuỗi hợp lệ, trả về {len(final_sequences)}")

        summary = {s['videoId']: s['matched_events'] for s in final_sequences}
        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        return jsonify({
            "results": final_sequences,
            "summary": sorted_summary,
            "events_query": parts,
            "max_gap_seconds": max_gap_seconds,
            "mode": "trake_temporal"
        })

    except ModelUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"Lỗi trong /search_trake_02: {e}")
        return jsonify({"error": str(e)}), 500
# (THÊM MỚI) API /search_trake_image (TRAKE.02 với nhiều ảnh)
@app.route('/search_trake_image', methods=['POST'])
def search_trake_image():
    try:
        # Lấy FormData trực tiếp (hỗ trợ multiple files)
        data = request.form.to_dict()
        top_k_final = int(data.get('top_k', 50))
        group_results = data.get('group', 'false').lower() == 'true'
        
        # Lấy multiple images
        image_files = request.files.getlist('image_files')  # Array of files
        if not image_files or len(image_files) < 2:
            return jsonify({"error": "Cần ít nhất 2 ảnh để tìm giao (TRAKE.02 Image)."}), 400
        
        # (CẬP NHẬT) Ngưỡng "common points" (cho phép thiếu 1 ảnh nếu có nhiều ảnh)
        min_common = max(2, len(image_files) - 1) if len(image_files) >= 3 else len(image_files)
        
        # (CẬP NHẬT) Nới khung hình (Frame Windowing)
        window_size = int(data.get('window_size', 5)) # Mặc định nới +/- 5 frames
        
        # n1 (số frame cần lấy cho mỗi ảnh)
        top_k_per_image = max(top_k_final * 100, 10000)
        
        print(f"[TRAKE.02 Image] Đang tìm kiếm {len(image_files)} ảnh, n1={top_k_per_image}, min_common={min_common}, window_size=±{window_size}")
        
        # Store frame occurrences
        # Key: (video_id, frame_n_str), Value: Dict[img_index -> max_score]
        frame_image_scores = collections.defaultdict(lambda: collections.defaultdict(float))
        # Key: (video_id, frame_n_str), Value: {path, videoId, pts_time}
        frame_info_cache = {}
        # Key: video_id, Value: count (for summary)
        summary_counter = collections.defaultdict(int)
        
        for img_index, file in enumerate(image_files):
                # Xử lý từng ảnh
                if file.filename == '': continue
                image = Image.open(io.BytesIO(file.read())).convert("RGB")

                query_vector = jina_text_encoder.encode_image(image)
                
                # Tìm n1 frames cho ảnh này
                distances, indices = jina_semantic_index.search(
                    query_vector, top_k_per_image
                )
                
                for i, dist in zip(indices[0], distances[0]):
                    original_path = image_records[int(i)]['path']
                    web_path, video_id, frame_n_str = get_web_path(original_path)
                    
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        video_meta = metadata_cache.get(video_id, {})
                        
                        # (CẬP NHẬT) Quét cửa sổ +/- window_size
                        for neighbor_n in get_neighbor_frame_ids(video_id, frame_n_int, window_size):
                            frame_key = (video_id, neighbor_n)

                            current_best = frame_image_scores[frame_key].get(img_index, -1000.0)
                            if float(dist) > current_best:
                                frame_image_scores[frame_key][img_index] = float(dist)

                            if frame_key not in frame_info_cache:
                                meta = video_meta[neighbor_n]
                                neighbor_web_path = meta.get('path')
                                if not neighbor_web_path:
                                    continue
                                pts_time = float(meta.get('pts_time', 0) or 0)
                                frame_info_cache[frame_key] = {
                                    "path": neighbor_web_path,
                                    "videoId": video_id,
                                    "pts_time": pts_time
                                }

        # Lọc kết quả dựa trên số "common images"
        all_results = []
        for frame_key, imgs_dict in frame_image_scores.items():
            common_count = len(imgs_dict)
            if common_count >= min_common:
                # Frame này match với ít nhất `min_common` ảnh
                info = frame_info_cache[frame_key].copy()
                info['sum_score'] = sum(imgs_dict.values())
                info['score'] = info['sum_score']
                info['common_count'] = common_count
                all_results.append(info)
                summary_counter[info['videoId']] += 1
        
        print(f"[TRAKE.02 Image] Tìm thấy {len(all_results)} frame chung (>= {min_common} ảnh)")
        
        # Sắp xếp và định dạng output (giống TRAKE.02 text)
        sorted_summary = dict(sorted(summary_counter.items(), key=lambda item: item[1], reverse=True))
        
        if group_results:
            grouped_results = {}
            for res in all_results:
                video_id = res['videoId']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)
            
            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                # Sắp xếp theo common_count giảm dần, sau đó pts_time
                sorted_items = sorted(items, key=lambda x: (-x['score'], x['pts_time']))
                final_grouped_results[video_id] = sorted_items[:top_k_final]
            
            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            # Sắp xếp theo common_count giảm dần, sau đó pts_time
            final_results = sorted(all_results, key=lambda x: (-x['score'], x['pts_time']))
            return jsonify({"results": final_results[:top_k_final], "summary": sorted_summary})
            
    except ModelUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"Lỗi trong /search_trake_image: {e}")
        return jsonify({"error": str(e)}), 500
# API /search_asr - tìm trực tiếp trên từng ASR segment bằng BM25.
@app.route('/search_asr', methods=['POST'])
def search_asr():
    try:
        data = request.get_json()
        if data is None:
             return jsonify({"error": "Request phải là JSON"}), 400
             
        query_text = data['query']
        top_k = int(data.get('top_k', 50))
        group_results = data.get('group', False)
        
        if not query_text:
            return jsonify({"results": [], "summary": {}})

        search_size = top_k * 5 if group_results else top_k * 2
        top_k_documents = asr_candidates(query_text, search_size)
        if not top_k_documents and not bm25_asr_index:
            return jsonify({"error": "Chưa có dữ liệu ASR (bm25_asr_index chưa khởi tạo)."}), 500

        # ASR hiện tại đã chia thành các segment dài, vì vậy trả từng segment
        # độc lập. Cơ chế stitch previous/current/next dành cho bộ ASR segment
        # ngắn trước đây đã được đưa vào backlog.
        final_results = []
        summary = {}

        for doc in top_k_documents:
            video_id = doc['video_id']
            final_doc = doc.copy()

            final_doc['watch_url'] = video_url_cache.get(video_id)

            target_start_time = final_doc['start']
            closest_frame_data = find_closest_keyframe(video_id, target_start_time)
            final_doc['frame_n']   = closest_frame_data.get('frame_n')
            final_doc['frame_idx'] = closest_frame_data.get('frame_idx')

            frame_n = final_doc['frame_n']
            final_doc['web_path'] = (
                get_frame_web_path(video_id, frame_n) if frame_n is not None else None
            )

            final_results.append(final_doc)
            summary[video_id] = summary.get(video_id, 0) + 1
        
        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in final_results:
                video_id = res['video_id']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)
            
            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                # Sắp xếp ASR theo thời gian bắt đầu
                sorted_items = sorted(items, key=lambda x: x['start'])
                final_grouped_results[video_id] = sorted_items[:top_k]
            
            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            final_results = sorted(final_results, key=lambda x: x['score'], reverse=True)[:top_k]
            return jsonify({"results": final_results, "summary": sorted_summary})

    except Exception as e:
        print(f"Lỗi trong /search_asr: {e}")
        return jsonify({"error": str(e)}), 500


# API /search_fusion - Gộp Jina Hybrid + OCR + ASR bằng RRF có trọng số.
# Mỗi khoảnh khắc được định danh bằng (video_id, frame_n) để gộp điểm giữa 3 nguồn có thang đo khác nhau
# (Jina Hybrid: RRF image/caption, OCR/ASR: điểm BM25 - không thể cộng trực tiếp).
@app.route('/search_fusion', methods=['POST'])
def search_fusion():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request phải là JSON"}), 400

        query_jina = data.get('query_jina', '').strip()
        query_ocr = data.get('query_ocr', '').strip()
        query_asr = data.get('query_asr', '').strip()
        if not query_jina and not query_ocr and not query_asr:
            return jsonify({"results": [], "summary": {}})

        # Trọng số thô (mặc định bằng nhau) - chỉ tỷ lệ tương đối giữa 3 số này mới ảnh hưởng thứ hạng RRF,
        # nên không cần chuẩn hoá về tổng=1 trước khi tính.
        weight_jina = float(data.get('weight_jina', 1.0))
        weight_ocr = float(data.get('weight_ocr', 1.0))
        weight_asr = float(data.get('weight_asr', 1.0))

        top_k = int(data.get('top_k', 50))
        group_results = data.get('group', False)
        pool_k = max(top_k * 5, 100)

        jina_ranked_keys, ocr_ranked_keys, asr_ranked_keys = [], [], []
        key_info = {}
        key_sources = collections.defaultdict(set)

        def register(key, web_path, video_id, pts_time, source):
            key_sources[key].add(source)
            if key not in key_info:
                key_info[key] = {
                    "path": web_path,
                    "videoId": video_id,
                    "pts_time": float(pts_time) if pts_time else 0.0
                }

        # --- 1. Nhánh Jina Hybrid (Jina image + Jina caption) ---
        if query_jina:
            try:
                batch1_jina_keys = []
                _, indices = search_semantic_text(
                    query_jina, "jina-hybrid", pool_k
                )
                for i in indices[0]:
                    i = int(i)
                    if i < 0:
                        continue
                    web_path, video_id, frame_n_str = get_web_path(image_records[int(i)]['path'])
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        key = (video_id, frame_n_int)
                        batch1_jina_keys.append(key)
                        meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                        register(key, web_path, video_id, meta.get('pts_time', 0) if meta else 0, "JINA_HYBRID")

                batch2_jina_keys = []
                if portable_image_index is not None or traffic_search_index is not None:
                    query_vector = _cached_semantic_query_vector(
                        query_jina, "jina-hybrid"
                    )
                    batch2_hits = search_supplemental_semantic(
                        query_jina,
                        "jina-hybrid",
                        query_vector,
                        top_k=pool_k,
                    )
                    for item in batch2_hits:
                        frame_idx = int(item["frame_idx"])
                        key = (item["videoId"], frame_idx)
                        batch2_jina_keys.append(key)
                        register(
                            key,
                            item["path"],
                            item["videoId"],
                            item["pts_time"],
                            "JINA_BATCH2",
                        )

                # Interleave both batches before this unified Jina branch is
                # fused with OCR/ASR, so Batch 2 does not count as an extra
                # modality and accidentally double weight_jina.
                for rank in range(max(len(batch1_jina_keys), len(batch2_jina_keys))):
                    if rank < len(batch1_jina_keys):
                        jina_ranked_keys.append(batch1_jina_keys[rank])
                    if rank < len(batch2_jina_keys):
                        jina_ranked_keys.append(batch2_jina_keys[rank])
            except Exception as e:
                print(f"[Fusion] Lỗi nhánh Jina Hybrid: {e}")

        # --- 2. Nhánh OCR (BM25 tự implement) ---
        if query_ocr:
            try:
                ocr_hits = ocr_candidates(query_ocr.lower(), pool_k)
                for score, original_path in ocr_hits:
                    web_path, video_id, frame_n_str = get_web_path(original_path)
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        key = (video_id, frame_n_int)
                        ocr_ranked_keys.append(key)
                        meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                        register(key, web_path, video_id, meta.get('pts_time', 0) if meta else 0, "OCR")
            except Exception as e:
                print(f"[Fusion] Lỗi nhánh OCR: {e}")

        # --- 3. Nhánh ASR (BM25 tự implement) ---
        if query_asr:
            try:
                asr_hits = asr_candidates(query_asr, pool_k)
                for src in asr_hits:
                    video_id = src.get("video_id", "")
                    if not video_id:
                        continue
                    closest = find_closest_keyframe(video_id, src.get("start", 0.0))
                    frame_n = closest.get('frame_n')
                    if frame_n is None:
                        continue
                    frame_n_int = int(frame_n)
                    web_path = get_frame_web_path(video_id, frame_n_int)
                    if not web_path:
                        continue
                    key = (video_id, frame_n_int)
                    asr_ranked_keys.append(key)
                    meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                    pts_time = meta.get('pts_time', 0) if meta else src.get("start", 0.0)
                    register(key, web_path, video_id, pts_time, "ASR")
            except Exception as e:
                print(f"[Fusion] Lỗi nhánh ASR: {e}")

        # --- 4. Gộp bằng RRF có trọng số ---
        branches = [
            (jina_ranked_keys, weight_jina),
            (ocr_ranked_keys, weight_ocr),
            (asr_ranked_keys, weight_asr),
        ]
        branches = [(lst, w) for lst, w in branches if lst]
        if not branches:
            return jsonify({"results": [], "summary": {}, "error": "Không có nhánh nào (Jina Hybrid/OCR/ASR) trả về kết quả."})

        ranked_lists = [lst for lst, _ in branches]
        weights = [w for _, w in branches]
        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)

        results = []
        summary = {}
        for key, score in fused:
            info = key_info.get(key)
            if not info:
                continue
            video_id = info["videoId"]
            results.append({
                "path": info["path"],
                "videoId": video_id,
                "score": float(score),
                "pts_time": info["pts_time"],
                "matched_by": sorted(key_sources[key])
            })
            summary[video_id] = summary.get(video_id, 0) + 1

        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in results:
                video_id = res['videoId']
                if video_id == "N/A":
                    continue
                grouped_results.setdefault(video_id, []).append(res)

            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                sorted_items = sorted(items, key=lambda x: x['pts_time'])
                final_grouped_results[video_id] = sorted_items[:top_k]

            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            return jsonify({"results": results[:top_k], "summary": sorted_summary})

    except Exception as e:
        print(f"Lỗi trong /search_fusion: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/search_traffic', methods=['POST'])
def search_traffic():
    """Batch 2 auto-discovered semantic search + optional detection reranking."""
    if traffic_search_index is None:
        return jsonify({"error": traffic_search_reason or "Traffic Search chưa sẵn sàng."}), 503
    try:
        payload = request.get_json(silent=True) or {}
        query_text = str(payload.get("query") or "").strip()
        if not query_text:
            return jsonify({"error": "Vui lòng nhập tình huống giao thông cần tìm."}), 400
        top_k = max(1, min(int(payload.get("top_k", 100)), 500))
        group_results = bool(payload.get("group", False))
        parsed_query = parse_traffic_query(query_text)
        model_query = expanded_query(query_text, parsed_query)
        query_vector = encode_semantic_query(model_query, "jina")
        pool_k = min(traffic_search_index.size, top_k * 5 if group_results else top_k)
        results = traffic_search_index.search(query_text, query_vector, top_k=pool_k)

        summary = {}
        for item in results:
            video_id = item["videoId"]
            summary[video_id] = summary.get(video_id, 0) + 1
        summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        response = {
            "results": results,
            "summary": summary,
            "mode": "traffic",
            "query_analysis": {
                "vehicles": parsed_query["vehicles"],
                "colors": parsed_query["colors"],
                "busy": parsed_query["busy"],
                "sparse": parsed_query["sparse"],
            },
        }
        if group_results:
            grouped = {}
            for item in results:
                grouped.setdefault(item["videoId"], []).append(item)
            for items in grouped.values():
                items.sort(key=lambda item: item["pts_time"])
            response["results"] = grouped
        return jsonify(response)
    except ModelUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"Lỗi trong /search_traffic: {exc}")
        return jsonify({"error": str(exc)}), 500


# (CẬP NHẬT) API /metadata
@app.route('/metadata', methods=['POST'])
def get_metadata():
    try:
        image_path = request.json['image_path']
        if portable_image_index is not None:
            portable_meta = portable_image_index.metadata_for_path(image_path)
            if portable_meta is not None:
                enrichment = (
                    traffic_search_index.metadata_for_video_frame(
                        portable_meta["video_id"], portable_meta["frame_idx"]
                    )
                    if traffic_search_index is not None else None
                )
                if enrichment is not None:
                    portable_meta["pts_time"] = float(enrichment["pts_time"])
                    portable_meta["caption"] = enrichment.get("caption", "")
                portable_meta["image_available"] = (
                    traffic_search_index.image_asset_for_video_frame(
                        portable_meta["video_id"], portable_meta["frame_idx"]
                    ) is not None
                )
                portable_meta.update(build_playback_info(
                    portable_meta["video_id"], portable_meta["pts_time"]
                ))
                return jsonify(portable_meta)
        if traffic_search_index is not None:
            traffic_meta = traffic_search_index.metadata_for_path(image_path)
            if traffic_meta is not None:
                traffic_meta.update(build_playback_info(
                    traffic_meta["video_id"], traffic_meta["pts_time"]
                ))
                return jsonify(traffic_meta)
        _, video_id, frame_id_str = get_web_path(image_path)
        if not frame_id_str or video_id == "N/A":
            raise ValueError(f"Invalid keyframe path: {image_path}")
        frame_id = int(frame_id_str)
        meta = dict(metadata_cache.get(video_id, {}).get(frame_id, {}))
        meta['n'] = frame_id
        meta.update(build_playback_info(video_id, meta.get('pts_time', 0)))
        return jsonify(meta)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# (CẬP NHẬT) API /neighbor_frames
@app.route('/neighbor_frames', methods=['POST'])
def get_neighbor_frames():
    try:
        payload = request.get_json() or {}
        image_path = payload['image_path']
        radius = max(1, min(int(payload.get('radius', 15)), 50))
        if portable_image_index is not None:
            portable_neighbors = portable_image_index.neighbors(image_path, radius)
            if portable_neighbors:
                return jsonify({"neighbors": portable_neighbors})
        if traffic_search_index is not None:
            traffic_neighbors = traffic_search_index.neighbors(image_path, radius)
            if traffic_neighbors:
                return jsonify({"neighbors": traffic_neighbors})
        _, video_id, frame_id_str = get_web_path(image_path)
        if not frame_id_str or video_id == "N/A":
            return jsonify({"neighbors": []})
        neighbor_ids = get_neighbor_frame_ids(video_id, int(frame_id_str), radius)
        neighbors = [get_frame_web_path(video_id, frame_id) for frame_id in neighbor_ids]
        return jsonify({"neighbors": [path for path in neighbors if path]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === (CẬP NHẬT) API ĐỂ LẤY BẢN ĐỒ THỜI GIAN KEYFRAME ===
@app.route('/get_keyframe_map', methods=['POST'])
def get_keyframe_map():
    try:
        video_id = request.json['video_id']
        if traffic_search_index is not None:
            traffic_map = traffic_search_index.keyframe_map(video_id)
            if traffic_map:
                return jsonify(traffic_map)
        if portable_image_index is not None:
            portable_map = portable_image_index.keyframe_map(video_id)
            if portable_map:
                return jsonify(portable_map)
        map_data = keyframe_time_cache.get(video_id)
        # (SỬA LỖI) Thêm check `if map_data`
        if map_data:
            return jsonify(map_data)
        else:
            return jsonify({"error": f"Map data not found for video_id: {video_id}"}), 404
    # (SỬA LỖI) Thụt lề khối 'except'
    except Exception as e: 
        print(f"Lỗi khi lấy bản đồ keyframe: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/dres/status', methods=['POST'])
def dres_status():
    """Read-only connectivity/evaluation check; never sends an answer."""
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "JSON request phải là object."}), 400
    result = check_evaluations(
        str(payload.get('session_id') or '').strip(),
        str(payload.get('evaluation_id') or '').strip(),
    )
    return jsonify(result), (200 if result['status'] == 'ok' else 400)


@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"status": "rejected", "message": "JSON request phải là object."}), 400
    result = submit_to_dres(
        str(payload.get('session_id') or '').strip(),
        str(payload.get('evaluation_id') or '').strip(),
        payload.get('answer_payload'),
    )
    # A transport failure after POST is uncertain; the browser must not auto-retry.
    http_status = 200 if result['status'] == 'accepted' else (409 if result['status'] == 'unknown' else 400)
    return jsonify(result), http_status


# --- VÒNG SƠ TUYỂN AIC26: resolve frame thật và xuất submission.zip ---
SUBMISSION_QUERY_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}-(kis|qa|trake)$",
    re.IGNORECASE,
)


def _submission_query_name(raw_name):
    name = str(raw_name or "").strip()
    if name.lower().endswith((".txt", ".csv")):
        name = name.rsplit(".", 1)[0]
    match = SUBMISSION_QUERY_ID_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(
            f"Tên query không hợp lệ: {name!r}; tên phải kết thúc bằng -kis, -qa hoặc -trake."
        )
    return name, match.group(1).lower()


@app.route('/submission/resolve_candidates', methods=['POST'])
def resolve_submission_candidates():
    """Map kết quả retrieval về frame_idx thật trước khi đưa vào CSV."""
    try:
        payload = request.get_json() or {}
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("candidates phải là một JSON array.")
        if len(candidates) > 1000:
            raise ValueError("Chỉ resolve tối đa 1.000 candidates mỗi request.")

        resolved = []
        errors = []
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                errors.append({"index": position, "error": "Candidate không phải object."})
                continue

            video_id = str(
                candidate.get("videoId") or candidate.get("video_id") or ""
            ).upper()
            frame_n = candidate.get("frame_n")
            path = candidate.get("path") or candidate.get("web_path")

            if path:
                portable_meta = (
                    portable_image_index.metadata_for_path(path)
                    if portable_image_index is not None else None
                )
                if portable_meta is not None:
                    video_id = portable_meta["video_id"]
                    frame_n = int(portable_meta["frame_idx"])
                    candidate.setdefault("frame_idx", int(portable_meta["frame_idx"]))
                else:
                    web_path, parsed_video_id, parsed_frame_n = get_web_path(path)
                    if web_path:
                        path = web_path
                    if parsed_video_id and parsed_video_id != "N/A":
                        video_id = parsed_video_id
                    if parsed_frame_n is not None:
                        frame_n = int(parsed_frame_n)

            meta = None
            if video_id in metadata_cache and frame_n is not None:
                meta = metadata_cache[video_id].get(int(frame_n))

            frame_idx = meta.get("frame_idx") if meta else candidate.get("frame_idx")
            if not video_id or frame_idx is None:
                errors.append({
                    "index": position,
                    "error": "Không map được video_id/frame_idx.",
                })
                continue

            resolved.append({
                "videoId": video_id,
                "frameIdx": int(frame_idx),
                "path": (meta or {}).get("path") or path,
                "score": float(candidate.get("score", 0) or 0),
            })

        return jsonify({"resolved": resolved, "errors": errors})
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/submission/neighbors', methods=['POST'])
def resolve_submission_neighbors():
    """Return keyframes around one or more pinned timestamps in the same videos."""
    try:
        payload = request.get_json() or {}
        anchors = payload.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError("Cần ít nhất một frame ghim để auto-fill.")
        if len(anchors) > 100:
            raise ValueError("Chỉ nhận tối đa 100 frame ghim.")

        time_border = float(payload.get("timeBorder", 30))
        if not math.isfinite(time_border) or time_border <= 0 or time_border > 600:
            raise ValueError("Biên thời gian phải lớn hơn 0 và không quá 600 giây.")
        limit = max(1, min(int(payload.get("limit", 1000)), 1000))

        results = []
        seen = set()
        resolved_anchors = []
        for position, anchor in enumerate(anchors, start=1):
            if not isinstance(anchor, dict):
                raise ValueError(f"Anchor {position} không hợp lệ.")
            video_id = str(anchor.get("videoId") or anchor.get("video_id") or "").upper()
            records = metadata_cache.get(video_id)
            portable_map = (
                portable_image_index.keyframe_map(video_id)
                if portable_image_index is not None else None
            )
            traffic_map = (
                traffic_search_index.keyframe_map(video_id)
                if traffic_search_index is not None else None
            )
            if not records and not portable_map and not traffic_map:
                raise ValueError(f"Không tìm thấy metadata của {video_id!r}.")

            raw_pts_time = anchor.get("ptsTime", anchor.get("pts_time"))
            target_time = None
            if raw_pts_time not in (None, ""):
                target_time = float(raw_pts_time)
                if not math.isfinite(target_time) or target_time < 0:
                    raise ValueError(f"Timestamp anchor {position} không hợp lệ.")

            raw_frame_idx = anchor.get("frameIdx", anchor.get("frame_idx"))
            if not records:
                if target_time is None:
                    if raw_frame_idx in (None, ""):
                        raise ValueError(f"Anchor {position} thiếu frame_idx/timestamp.")
                    closest = (
                        traffic_search_index.nearest_video_frame(
                            video_id, frame_idx=int(raw_frame_idx)
                        ) if traffic_map else portable_image_index.nearest_video_frame(
                            video_id, frame_idx=int(raw_frame_idx)
                        )
                    )
                    target_time = float(closest["pts_time"])
                nearby = (
                    traffic_search_index.results_around_time(
                        video_id, target_time, time_border, limit - len(results)
                    ) if traffic_map else portable_image_index.results_around_time(
                        video_id, target_time, time_border, limit - len(results)
                    )
                )
                canonical_video_id = nearby[0]["videoId"] if nearby else video_id
                resolved_anchors.append({"videoId": canonical_video_id, "ptsTime": target_time})
                for item in nearby:
                    candidate_key = normalized_frame_key(item["videoId"], item["frame_idx"])
                    if candidate_key in seen:
                        continue
                    seen.add(candidate_key)
                    results.append({
                        "videoId": item["videoId"],
                        "frameIdx": int(item["frame_idx"]),
                        "frame_n": int(item["frame_idx"]),
                        "ptsTime": float(item["pts_time"]),
                        "path": item["path"],
                        "distanceSeconds": abs(float(item["pts_time"]) - target_time),
                    })
                if len(results) >= limit:
                    break
                continue

            if target_time is None:
                if raw_frame_idx in (None, ""):
                    raise ValueError(f"Anchor {position} thiếu frame_idx/timestamp.")
                frame_idx = int(raw_frame_idx)
                fps = keyframe_time_cache.get(video_id, {}).get("fps")
                if fps and float(fps) > 0:
                    target_time = frame_idx / float(fps)
                else:
                    closest = min(
                        records.values(),
                        key=lambda record: abs(int(record.get("frame_idx", 0)) - frame_idx),
                    )
                    target_time = float(closest.get("pts_time", 0) or 0)

            nearby = [
                record for record in records.values()
                if abs(float(record.get("pts_time", 0) or 0) - target_time) <= time_border
            ]
            nearby.sort(key=lambda record: (
                abs(float(record.get("pts_time", 0) or 0) - target_time),
                float(record.get("pts_time", 0) or 0),
            ))
            resolved_anchors.append({"videoId": video_id, "ptsTime": target_time})
            for record in nearby:
                candidate_key = (video_id, int(record["frame_idx"]))
                if candidate_key in seen:
                    continue
                seen.add(candidate_key)
                pts_time = float(record.get("pts_time", 0) or 0)
                results.append({
                    "videoId": video_id,
                    "frameIdx": int(record["frame_idx"]),
                    "frame_n": int(record["frame_id"]),
                    "ptsTime": pts_time,
                    "path": record.get("path") or "",
                    "distanceSeconds": abs(pts_time - target_time),
                })
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        return jsonify({
            "results": results,
            "anchors": resolved_anchors,
            "timeBorder": time_border,
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/submission/playback', methods=['POST'])
def resolve_submission_playback():
    """Return the video URL nearest to a submitted frame_idx."""
    try:
        payload = request.get_json() or {}
        video_id = str(payload.get("videoId") or "").upper()
        frame_idx = int(payload.get("frameIdx"))
        video_records = metadata_cache.get(video_id)

        raw_pts_time = payload.get("ptsTime", payload.get("pts_time"))
        if raw_pts_time not in (None, ""):
            pts_time = float(raw_pts_time)
        else:
            fps = keyframe_time_cache.get(video_id, {}).get("fps")
            pts_time = frame_idx / float(fps) if fps and float(fps) > 0 else None
        if pts_time is not None and (not math.isfinite(pts_time) or pts_time < 0):
            raise ValueError("Timestamp không hợp lệ.")

        if not video_records:
            traffic_frame = (
                traffic_search_index.nearest_video_frame(video_id, frame_idx, pts_time)
                if traffic_search_index is not None
                else None
            )
            if traffic_frame is None:
                traffic_frame = (
                portable_image_index.nearest_video_frame(video_id, frame_idx, pts_time)
                if portable_image_index is not None
                else None
                )
            if traffic_frame is None:
                raise ValueError(f"Không tìm thấy video {video_id!r}.")
            resolved_time = float(traffic_frame["pts_time"])
            playback_info = build_playback_info(video_id, resolved_time)
            return jsonify({
                "videoId": video_id,
                "requestedFrameIdx": frame_idx,
                "frameIdx": frame_idx,
                "keyframeFrameIdx": int(traffic_frame["frame_idx"]),
                "pts_time": resolved_time,
                "path": traffic_frame["path"],
                **playback_info,
            })

        if pts_time is None:
            closest = min(
                video_records.values(),
                key=lambda record: abs(int(record.get("frame_idx", 0)) - frame_idx),
            )
            pts_time = float(closest.get("pts_time", 0) or 0)
        else:
            closest = min(
                video_records.values(),
                key=lambda record: abs(float(record.get("pts_time", 0) or 0) - pts_time),
            )
        watch_url = video_url_cache.get(video_id)
        if watch_url:
            separator = '&' if '?' in watch_url else '?'
            playback_url = f"{watch_url}{separator}t={int(pts_time)}s"
        else:
            playback_url = None

        return jsonify({
            "videoId": video_id,
            "requestedFrameIdx": frame_idx,
            "frameIdx": frame_idx,
            "keyframeFrameIdx": int(closest.get("frame_idx", 0)),
            "pts_time": pts_time,
            "path": closest.get("path"),
            "playback_url": playback_url,
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/submission/export', methods=['POST'])
def export_preliminary_submission():
    """Validate ordered rows and return a ZIP containing submission/*.csv."""
    try:
        payload = request.get_json() or {}
        queries = payload.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError("Chưa có query nào để xuất.")

        archive_buffer = io.BytesIO()
        used_names = set()
        with zipfile.ZipFile(
            archive_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for query in queries:
                if not isinstance(query, dict):
                    raise ValueError("Mỗi query phải là một object.")
                query_name, inferred_type = _submission_query_name(query.get("id"))
                query_type = str(query.get("type") or inferred_type).strip().lower()
                if query_type != inferred_type:
                    raise ValueError(
                        f"Loại {query_type!r} không khớp tên file {query_name!r}."
                    )
                normalized_name = query_name.lower()
                if normalized_name in used_names:
                    raise ValueError(f"Trùng query: {query_name}.")
                used_names.add(normalized_name)

                rows = query.get("rows")
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"{query_name} chưa có kết quả.")
                if len(rows) > 100:
                    raise ValueError(f"{query_name} vượt quá 100 dòng.")

                event_count = int(query.get("eventCount") or 0)
                output = io.StringIO(newline="")
                writer = csv.writer(output, lineterminator="\n")
                seen_rows = set()
                for row_number, row in enumerate(rows, start=1):
                    if not isinstance(row, dict):
                        raise ValueError(f"{query_name} dòng {row_number} không hợp lệ.")
                    video_id = str(row.get("videoId") or "").upper()
                    if not re.fullmatch(r"L\d{2}_V\d+", video_id):
                        raise ValueError(
                            f"{query_name} dòng {row_number}: video ID không hợp lệ."
                        )

                    if query_type in {"kis", "qa"}:
                        frame_idx = int(row.get("frameIdx"))
                        if frame_idx < 0:
                            raise ValueError(
                                f"{query_name} dòng {row_number}: frame_idx phải không âm."
                            )
                        csv_row = [video_id, frame_idx]
                        if query_type == "qa":
                            answer = str(row.get("answer") or "").strip()
                            if not answer:
                                raise ValueError(
                                    f"{query_name} dòng {row_number}: thiếu answer."
                                )
                            if len(answer) > 100:
                                raise ValueError(
                                    f"{query_name} dòng {row_number}: answer vượt 100 ký tự."
                                )
                            csv_row.append(answer)
                    else:
                        frame_indices = row.get("frameIndices")
                        if not isinstance(frame_indices, list):
                            raise ValueError(
                                f"{query_name} dòng {row_number}: thiếu danh sách frame TRAKE."
                            )
                        frame_indices = [int(value) for value in frame_indices]
                        if any(value < 0 for value in frame_indices):
                            raise ValueError(
                                f"{query_name} dòng {row_number}: frame TRAKE phải không âm."
                            )
                        if event_count < 2 or len(frame_indices) != event_count:
                            raise ValueError(
                                f"{query_name} dòng {row_number}: cần đúng {event_count} frames."
                            )
                        if any(
                            current <= previous
                            for previous, current in zip(frame_indices, frame_indices[1:])
                        ):
                            raise ValueError(
                                f"{query_name} dòng {row_number}: frames phải tăng theo thời gian."
                            )
                        csv_row = [video_id, *frame_indices]

                    row_key = tuple(csv_row)
                    if row_key in seen_rows:
                        raise ValueError(f"{query_name} có dòng trùng: {csv_row}.")
                    seen_rows.add(row_key)
                    writer.writerow(csv_row)

                archive.writestr(
                    f"submission/{query_name}.csv",
                    output.getvalue().encode("utf-8"),
                )

        archive_buffer.seek(0)
        return send_file(
            archive_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name="submission.zip",
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

# --- Các hàm phục vụ file tĩnh ---
PUBLIC_STATIC_FILES = {
    "style.css",
    "script.js",
    "share_client.css",
    "share_client.js",
    "logo_wud.jpg",
    "submission-builder.css",
    "submission-builder.js",
    "submission-store.js",
}


@app.route('/health', methods=['GET'])
def health():
    """Readiness check without forcing either lazy semantic model to load."""
    jina_available, jina_reason = jina_text_encoder.availability()
    hybrid_available = jina_available and jina_caption_index is not None
    return jsonify({
        "status": "ok" if hybrid_available else "degraded",
        "device": device,
        "records": len(image_records),
        "local_videos": {
            "available": bool(local_video_index),
            "count": len(local_video_index),
            "directory": str(VIDEOS_DIR),
        },
        "traffic": {
            "available": traffic_search_index is not None,
            "reason": traffic_search_reason,
            "frames": traffic_search_index.size if traffic_search_index is not None else 0,
            "videos": traffic_search_index.video_count if traffic_search_index is not None else 0,
            "shards": (
                traffic_search_index.shard_names
                if traffic_search_index is not None else []
            ),
            "detection_frames": (
                traffic_search_index.detection_count
                if traffic_search_index is not None else 0
            ),
            "keyframes_available": (
                traffic_search_index.keyframes_available
                if traffic_search_index is not None else False
            ),
            "maps_available": (
                traffic_search_index.maps_available
                if traffic_search_index is not None else False
            ),
        },
        "portable_images": {
            "available": portable_image_index is not None,
            "reason": portable_image_reason,
            "frames": portable_image_index.ntotal if portable_image_index is not None else 0,
            "videos": portable_image_index.video_count if portable_image_index is not None else 0,
            "packages": (
                portable_image_index.package_names
                if portable_image_index is not None else []
            ),
        },
        "jina": {"available": jina_available, "reason": jina_reason},
        "jina_hybrid": {
            "available": hybrid_available,
            "reason": jina_caption_index_reason if jina_available else jina_reason,
        },
        "ocr": {"available": bm25_ocr_index is not None},
        "asr_for_fusion": {"available": bm25_asr_index is not None},
        "auto_crop": {"available": yolo_model is not None},
    })


@app.route('/')
def serve_index(): return send_from_directory(str(BASE_DIR), 'index.html')
@app.route('/submission-builder')
def serve_submission_builder():
    return send_from_directory(str(BASE_DIR), 'submission-builder.html')
@app.route('/videos/<video_id>')
def serve_local_video(video_id):
    asset = next(
        (
            local_video_index.get(alias)
            for alias in video_id_aliases(video_id)
            if local_video_index.get(alias) is not None
        ),
        None,
    )
    if asset is None:
        abort(404)
    asset_path = asset["path"]
    if not asset_path.is_file():
        abort(404)
    if asset["kind"] == "file":
        # conditional=True enables byte-range responses so browser seeking works.
        return send_file(str(asset_path), conditional=True)

    total_size = int(asset["size"])
    range_header = request.headers.get("Range", "").strip()
    start, end, status = 0, max(0, total_size - 1), 200
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
        if match is None or (not match.group(1) and not match.group(2)):
            return Response(status=416, headers={"Content-Range": f"bytes */{total_size}"})
        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else total_size - 1
        else:
            suffix_length = int(match.group(2))
            start = max(0, total_size - suffix_length)
            end = total_size - 1
        if start >= total_size or start > end:
            return Response(status=416, headers={"Content-Range": f"bytes */{total_size}"})
        end = min(end, total_size - 1)
        status = 206

    length = max(0, end - start + 1)

    def generate_zip_member():
        with zipfile.ZipFile(asset_path) as archive:
            with archive.open(asset["member"], "r") as stream:
                stream.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Cache-Control": "private, max-age=3600",
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"
    return Response(
        generate_zip_member(),
        status=status,
        mimetype="video/mp4",
        headers=headers,
        direct_passthrough=True,
    )
@app.route('/batch2-keyframes/<path:path>')
def serve_batch2_keyframes(path):
    logical_path = f"/batch2-keyframes/{path}"
    asset = None
    portable_meta = (
        portable_image_index.metadata_for_path(logical_path)
        if portable_image_index is not None else None
    )
    if portable_meta is not None and traffic_search_index is not None:
        asset = traffic_search_index.image_asset_for_video_frame(
            portable_meta["video_id"], portable_meta["frame_idx"]
        )
    if asset is None and traffic_search_index is not None:
        asset = traffic_search_index.image_asset_for_path(logical_path)
    if asset is not None:
        if asset["kind"] == "file":
            return send_file(
                str(asset["path"]),
                mimetype=asset["mimetype"],
                conditional=True,
            )
        return Response(
            asset["data"],
            mimetype=asset["mimetype"],
            headers={"Cache-Control": "private, max-age=3600"},
        )

    # Embedding-only smoke mode: preserve the logical image URL so metadata
    # lookup/click still works, but render a useful card instead of a broken img.
    parts = Path(path).parts
    video_id = parts[-2] if len(parts) >= 2 else "Batch 2"
    frame_name = Path(parts[-1]).stem if parts else "?"
    safe_video_id = html_escape(video_id)
    safe_frame_name = html_escape(frame_name)
    placeholder = f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
      <rect width="640" height="360" fill="#172635"/>
      <rect x="28" y="28" width="584" height="304" rx="18" fill="#22394d" stroke="#3d5b70"/>
      <text x="50%" y="43%" text-anchor="middle" fill="#79d4ff" font-family="Arial,sans-serif" font-size="30" font-weight="700">{safe_video_id}</text>
      <text x="50%" y="56%" text-anchor="middle" fill="#ffffff" font-family="Arial,sans-serif" font-size="22">frame {safe_frame_name}</text>
      <text x="50%" y="70%" text-anchor="middle" fill="#a9bac7" font-family="Arial,sans-serif" font-size="16">Embedding-only · chưa tải keyframe</text>
    </svg>'''
    return Response(
        placeholder,
        mimetype="image/svg+xml",
        headers={"Cache-Control": "private, max-age=300"},
    )
@app.route('/<path:path>')
def serve_static(path):
    if path not in PUBLIC_STATIC_FILES:
        abort(404)
    return send_from_directory(str(BASE_DIR), path)
@app.route('/Keyframes/<path:path>')
def serve_keyframes(path): return send_from_directory(str(KEYFRAMES_DIR), path)

# --- CHẠY APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
