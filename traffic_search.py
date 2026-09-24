"""Collection-based semantic retrieval with optional detection and visual assets.

Every directory containing ``caption_embeddings.npy`` and
``caption_mapping.csv`` below the configured caption root becomes one shard.
Detection parquet files, keyframe folders/ZIPs and metadata JSON enrich those
shards but are not required for search or submission tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import mimetypes
import os
import re
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

import numpy as np
import polars as pl


VEHICLE_ALIASES = {
    "car": ("o to", "xe hoi", "car", "cars", "sedan", "suv", "van", "hatchback"),
    "motorcycle": ("xe may", "mo to", "motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter", "scooters"),
    "bus": ("xe buyt", "bus", "buses"),
    "truck": ("xe tai", "truck", "trucks", "lorry", "lorries"),
    "bicycle": ("xe dap", "bicycle", "bicycles", "cyclist", "cyclists"),
}

COLOR_ALIASES = {
    "black": ("den", "black"), "white": ("trang", "white"),
    "red": ("do", "red"),
    "blue": ("xanh duong", "xanh da troi", "blue", "teal", "navy"),
    "green": ("xanh la", "green"),
    "yellow_gold": ("yellow", "gold", "golden"),
    "orange": ("cam", "orange"),
    "gray_silver": ("xam", "ghi", "bac", "gray", "grey", "silver"),
    "brown": ("nau", "brown", "beige"), "pink": ("hong", "pink"),
    "purple": ("purple", "violet"),
}
COLOR_CAPTION_TERMS = {
    name: tuple(alias for alias in aliases if alias.isascii() and " " not in alias)
    for name, aliases in COLOR_ALIASES.items()
}
BUSY_TERMS = ("dong xe", "nhieu xe", "ket xe", "un tac", "busy", "crowded", "congestion", "heavy traffic")
SPARSE_TERMS = ("vang xe", "it xe", "duong vang", "empty road", "light traffic", "few vehicles")
DETECTION_COLUMNS = {
    "car": "det_car_count", "motorcycle": "det_motorcycle_count",
    "bus": "det_bus_count", "truck": "det_truck_count",
    "bicycle": "det_bicycle_count",
}
# Lxx dùng index image/caption đã căn theo metadata toàn cục; loader này chỉ nhận
# các collection M/N/S dù tất cả cùng dùng một artifact root.
VIDEO_ID_RE = re.compile(r"^[NMS]\d{2,3}[-_]V\d+$", re.IGNORECASE)
FRAME_NUMBER_RE = re.compile(r"(\d+)$")
IMAGE_SUFFIXES = {".webp", ".jpg", ".jpeg", ".png"}
LEGACY_BATCH_DIR_RE = re.compile(r"^L\d{2}$", re.IGNORECASE)
COMMON_FPS = np.asarray((23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0))


def normalize_text(value: str) -> str:
    value = str(value or "").casefold().replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def parse_traffic_query(query: str) -> dict:
    normalized = normalize_text(query)
    vehicles = [name for name, aliases in VEHICLE_ALIASES.items()
                if any(_contains_phrase(normalized, alias) for alias in aliases)]
    colors = [name for name, aliases in COLOR_ALIASES.items()
              if any(_contains_phrase(normalized, alias) for alias in aliases)]
    # Accent removal makes tìm/tím and vắng/vàng collide.
    if _contains_phrase(normalized, "mau tim") and "purple" not in colors:
        colors.append("purple")
    if _contains_phrase(normalized, "mau vang") and "yellow_gold" not in colors:
        colors.append("yellow_gold")
    if _contains_phrase(normalized, "xanh") and "blue" not in colors and "green" not in colors:
        colors.append("blue")
    return {
        "normalized": normalized, "vehicles": vehicles, "colors": colors,
        "busy": any(_contains_phrase(normalized, term) for term in BUSY_TERMS),
        "sparse": any(_contains_phrase(normalized, term) for term in SPARSE_TERMS),
    }


def expanded_query(query: str, parsed: dict) -> str:
    english = [color.replace("_", " ") for color in parsed["colors"]]
    english.extend(parsed["vehicles"])
    if parsed["busy"]:
        english.extend(("busy", "heavy traffic"))
    if parsed["sparse"]:
        english.extend(("empty road", "few vehicles"))
    return f"{query}. Traffic scene: {' '.join(english)}" if english else query


def merge_ranked_batches(batch1_results, batch2_results, rrf_k=60.0):
    """Merge disjoint Batch 1/2 rankings without comparing raw score scales."""
    merged = []
    for source_results in (batch1_results, batch2_results):
        for rank, item in enumerate(source_results):
            merged_item = dict(item)
            merged_item["source_score"] = float(item.get("score", 0.0))
            merged_item["score"] = 1.0 / (float(rrf_k) + rank + 1.0)
            merged.append(merged_item)
    return sorted(
        merged,
        key=lambda item: (-item["score"], item.get("batch") != "batch1"),
    )


def _near_pair(caption: str, color_terms: tuple[str, ...], vehicle_terms: tuple[str, ...]) -> bool:
    words = re.findall(r"[a-z0-9]+", caption.casefold())
    colors = [i for i, word in enumerate(words) if word in color_terms]
    vehicles = [i for i, word in enumerate(words) if word in vehicle_terms]
    return any(abs(left - right) <= 4 for left in colors for right in vehicles)


def _artifact_key(value: str | Path) -> str:
    name = Path(value).stem.casefold()
    name = re.sub(r"_metadata$", "", name)
    name = re.sub(r"^(videos?|keyframes)[_-]", "", name)
    name = re.sub(r"_1fps$", "", name)
    # Một số artifact BTC dùng range N041-N50, trong khi folder embedding đã
    # chuẩn hóa thành N041-N050. Chuẩn hóa padding để enrichment cũ và shard mới
    # vẫn ghép đúng với nhau.
    def normalize_range(match):
        prefix = match.group(1)
        width = 3 if prefix == "n" else 2
        return (
            f"{prefix}{int(match.group(2)):0{width}d}-"
            f"{prefix}{int(match.group(3)):0{width}d}"
        )

    name = re.sub(r"\b([mns])(\d+)-\1(\d+)\b", normalize_range, name)
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def _frame_number(filename: str) -> int | None:
    match = FRAME_NUMBER_RE.search(Path(filename).stem)
    return int(match.group(1)) if match else None


def _video_frame_from_path(value: str) -> tuple[str, int] | None:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    video_id = next((p for p in reversed(parts[:-1]) if VIDEO_ID_RE.fullmatch(p)), None)
    frame_idx = _frame_number(parts[-1]) if parts else None
    return (video_id.upper(), frame_idx) if video_id is not None and frame_idx is not None else None


class TrafficSearchIndex:
    """Auto-discovered M/N/S caption index with optional enrichments."""

    DETECTION_FIELDS = (
        "member_path", "video_id", "frame_idx", "det_person_count",
        "det_traffic_light_count", "det_car_count", "det_motorcycle_count",
        "det_bus_count", "det_truck_count", "det_bicycle_count",
        "vehicle_count", "seg_road", "seg_person", "seg_vehicle_ratio",
    )

    def __init__(self, caption_dir, detection_path, keyframes_dir, map_dir,
                 media_info_dir=None, metadata_dir=None, cache_dir=None, search_dims=128,
                 web_prefix="/batch2-keyframes"):
        self.caption_root = Path(caption_dir)
        self.detection_root = Path(detection_path) if detection_path else None
        self.keyframes_dir = Path(keyframes_dir)
        self.map_dir = Path(map_dir) if map_dir else None
        self.media_info_dir = Path(media_info_dir) if media_info_dir else None
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.search_dims = max(32, min(int(search_dims), 1024))
        if self.cache_dir is None:
            self.search_dims = 1024
        self.web_prefix = "/" + web_prefix.strip("/")

        caption_shards = self._discover_caption_shards()
        detection_by_key = {_artifact_key(p): p for p in self._discover_detection_paths()}
        exact_timestamps, map_fps = self._load_keyframe_maps()
        metadata_timestamps, metadata_fps, metadata_urls = self._load_keyframe_metadata()
        exact_timestamps.update(metadata_timestamps)
        map_fps.update(metadata_fps)
        legacy_media_info = self._load_media_info()
        # Metadata OCR là nguồn canonical. Media-info cũ chỉ bù URL còn thiếu
        # trong giai đoạn migration và không được ghi đè metadata mới.
        self.video_url_by_id = {
            video_id: str(info["watch_url"])
            for video_id, info in legacy_media_info.items()
            if info.get("watch_url")
        }
        self.video_url_by_id.update(metadata_urls)

        self.embedding_shards, self.shard_names = [], []
        relative_paths, web_paths, video_ids, frame_indices, captions = [], [], [], [], []
        detection_available, vehicle_counts = [], []
        person_counts, light_counts = [], []
        seg_road, seg_person, seg_vehicle_ratio = [], [], []
        det_counts = {name: [] for name in DETECTION_COLUMNS}
        offset = 0

        for shard_dir, mapping_path, embedding_path in caption_shards:
            embeddings = np.load(embedding_path, mmap_mode="r")
            if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
                raise ValueError(f"Caption embedding shape không hợp lệ: {embedding_path} {embeddings.shape}")
            detector_path = detection_by_key.get(_artifact_key(shard_dir.name))
            by_path, by_frame = self._load_detection_lookup(detector_path)
            shard_name, row_count = shard_dir.name, 0
            with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
                for mapping in csv.DictReader(stream):
                    relative_path = PurePosixPath(str(mapping.get("relative_path") or "").replace("\\", "/")).as_posix()
                    parsed = _video_frame_from_path(relative_path)
                    if parsed is None:
                        parent = str(mapping.get("parent_path") or "").replace("\\", "/")
                        parsed = _video_frame_from_path(f"{parent}/{mapping.get('frame_name', '')}")
                    if parsed is None:
                        raise ValueError(f"Không đọc được video/frame từ mapping: {relative_path}")
                    video_id, frame_idx = parsed
                    detection = by_path.get(relative_path) or by_frame.get((video_id, frame_idx))
                    has_detection, detection = detection is not None, detection or {}
                    frame_name = str(mapping.get("frame_name") or PurePosixPath(relative_path).name)
                    web_path = (f"{self.web_prefix}/{quote(shard_name, safe='')}/"
                                f"{quote(video_id, safe='-_')}/{quote(frame_name, safe='._-')}")
                    relative_paths.append(relative_path)
                    web_paths.append(web_path)
                    video_ids.append(video_id)
                    frame_indices.append(frame_idx)
                    captions.append(str(mapping.get("caption") or "").strip())
                    detection_available.append(has_detection)
                    vehicle_counts.append(int(detection.get("vehicle_count") or 0))
                    person_counts.append(int(detection.get("det_person_count") or 0))
                    light_counts.append(int(detection.get("det_traffic_light_count") or 0))
                    seg_road.append(float(detection.get("seg_road") or 0.0))
                    seg_person.append(float(detection.get("seg_person") or 0.0))
                    seg_vehicle_ratio.append(float(detection.get("seg_vehicle_ratio") or 0.0))
                    for name, column in DETECTION_COLUMNS.items():
                        det_counts[name].append(int(detection.get(column) or 0))
                    row_count += 1
            if row_count != embeddings.shape[0]:
                raise ValueError(f"Caption mapping/embedding lệch dòng tại {shard_name}: {row_count} != {embeddings.shape[0]}")
            search_embeddings = self._load_search_embeddings(
                embeddings, embedding_path, shard_name
            )
            self.embedding_shards.append((offset, offset + row_count, search_embeddings))
            self.shard_names.append(shard_name)
            offset += row_count

        self.relative_paths, self.web_paths = relative_paths, web_paths
        self.video_ids, self.captions = video_ids, captions
        self.caption_lower = [value.casefold() for value in captions]
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.detection_available = np.asarray(detection_available, dtype=bool)
        self.vehicle_counts = np.asarray(vehicle_counts, dtype=np.int32)
        self.person_counts = np.asarray(person_counts, dtype=np.int32)
        self.traffic_light_counts = np.asarray(light_counts, dtype=np.int32)
        self.seg_road = np.asarray(seg_road, dtype=np.float32)
        self.seg_person = np.asarray(seg_person, dtype=np.float32)
        self.seg_vehicle_ratio = np.asarray(seg_vehicle_ratio, dtype=np.float32)
        self.det_counts = {name: np.asarray(values, dtype=np.int16) for name, values in det_counts.items()}

        self.rows_by_video = {}
        for index, video_id in enumerate(video_ids):
            self.rows_by_video.setdefault(video_id, []).append(index)
        self.fps_by_video = self._infer_fps(map_fps)
        self.timestamps = np.empty(self.size, dtype=np.float64)
        for index, (video_id, frame_idx) in enumerate(zip(video_ids, self.frame_indices)):
            timestamp = exact_timestamps.get((video_id, int(frame_idx)))
            if timestamp is None:
                fps = self.fps_by_video.get(video_id, 25.0)
                timestamp = float(frame_idx) / fps if fps > 0 else 0.0
            self.timestamps[index] = max(0.0, float(timestamp))
        for video_id, indices in self.rows_by_video.items():
            indices.sort(key=lambda row: (self.timestamps[row], self.frame_indices[row]))

        self.row_by_web_path = {path: index for index, path in enumerate(web_paths)}
        image_assets = self._discover_image_assets()
        assets_by_video = {}
        for (video_id, asset_frame), asset in image_assets.items():
            assets_by_video.setdefault(video_id, []).append((asset_frame, asset))
        for values in assets_by_video.values():
            values.sort(key=lambda item: item[0])
        self.assets_by_row, self.image_available = [], np.zeros(self.size, dtype=bool)
        self.assets_by_row = [None] * self.size
        for video_id, rows in self.rows_by_video.items():
            available = assets_by_video.get(video_id, [])
            mapping_frames = {int(self.frame_indices[row]) for row in rows}
            asset_frames = {frame for frame, _ in available}
            # Some official ZIPs name images 000000, 000001, ... while the
            # embedding mapping keeps original frame ids 0, 30, 60, ... . If
            # both sides have the same number of rows, align chronologically.
            sequential_assets = (
                len(available) == len(rows) and asset_frames != mapping_frames
            )
            for position, row in enumerate(rows):
                if sequential_assets:
                    asset = available[position][1]
                else:
                    asset = image_assets.get((video_id, int(self.frame_indices[row])))
                self.assets_by_row[row] = asset
                self.image_available[row] = asset is not None
        self.keyframes_available = bool(image_assets)
        self.maps_available = bool(exact_timestamps)

    def _discover_caption_shards(self):
        if not self.caption_root.is_dir():
            raise FileNotFoundError(self.caption_root)
        paths = list(self.caption_root.rglob("caption_embeddings.npy"))
        shards = [
            (p.parent, p.with_name("caption_mapping.csv"), p)
            for p in sorted(set(paths))
            if p.with_name("caption_mapping.csv").is_file()
            and not LEGACY_BATCH_DIR_RE.fullmatch(p.parent.name)
        ]
        if not shards:
            raise FileNotFoundError(f"Không tìm thấy cặp caption_embeddings.npy + caption_mapping.csv trong {self.caption_root}")
        return shards

    def _discover_detection_paths(self):
        root = self.detection_root
        if root is None or not root.exists():
            return []
        if root.is_file():
            return [root] if root.suffix.casefold() == ".parquet" else []
        return sorted(root.rglob("*.parquet"))

    def _load_search_embeddings(self, embeddings, source_path, shard_name):
        """Use a compact normalized prefix cache for low-latency smoke tests."""
        if self.search_dims >= embeddings.shape[1] or self.cache_dir is None:
            return embeddings
        stat = source_path.stat()
        fingerprint = hashlib.sha1(
            f"{source_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{self.search_dims}".encode()
        ).hexdigest()[:12]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", shard_name)
        cache_path = self.cache_dir / f"{safe_name}-{fingerprint}-d{self.search_dims}.npy"
        if cache_path.is_file():
            cached = np.load(cache_path, mmap_mode="r")
            if cached.shape == (embeddings.shape[0], self.search_dims):
                return cached
            cache_path.unlink(missing_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp.npy")
        target = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=(embeddings.shape[0], self.search_dims),
        )
        for start in range(0, embeddings.shape[0], 8192):
            end = min(start + 8192, embeddings.shape[0])
            block = np.asarray(embeddings[start:end, :self.search_dims], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            target[start:end] = block / np.maximum(norms, 1e-12)
        target.flush()
        del target
        temporary_path.replace(cache_path)
        return np.load(cache_path, mmap_mode="r")

    def _load_detection_lookup(self, path):
        if path is None:
            return {}, {}
        schema = set(pl.scan_parquet(path).collect_schema().names())
        if not {"member_path", "video_id", "frame_idx"}.issubset(schema):
            return {}, {}
        table = pl.read_parquet(path, columns=[name for name in self.DETECTION_FIELDS if name in schema])
        by_path, by_frame = {}, {}
        for row in table.iter_rows(named=True):
            member = PurePosixPath(str(row["member_path"]).replace("\\", "/")).as_posix()
            key = (str(row["video_id"]).upper(), int(row["frame_idx"]))
            by_path[member], by_frame[key] = row, row
        return by_path, by_frame

    def _load_keyframe_maps(self):
        timestamps, fps_by_video = {}, {}
        root = self.map_dir
        if root is None or not root.is_dir():
            return timestamps, fps_by_video
        map_paths = []
        for current, directories, filenames in os.walk(root):
            directories[:] = [name for name in directories if not LEGACY_BATCH_DIR_RE.fullmatch(name)]
            map_paths.extend(Path(current) / name for name in filenames if name.casefold().endswith(".csv"))
        for path in sorted(map_paths):
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"frame_id", "timestamp"}.issubset(set(reader.fieldnames or ())):
                    continue
                rows = list(reader)
            video_id = path.stem.upper()
            if not rows or not VIDEO_ID_RE.fullmatch(video_id):
                continue
            fps_values = {round(float(row["fps"]), 6) for row in rows if row.get("fps")}
            if len(fps_values) == 1:
                fps_by_video[video_id] = next(iter(fps_values))
            for row in rows:
                timestamps[(video_id, int(row["frame_id"]))] = float(row["timestamp"])
        return timestamps, fps_by_video

    def _load_media_info(self):
        root, info = self.media_info_dir, {}
        if root is None or not root.exists():
            return info
        paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in paths:
            video_id = path.stem.upper()
            if not VIDEO_ID_RE.fullmatch(video_id):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            info[video_id] = payload if isinstance(payload, dict) else {}
        return info

    def _load_keyframe_metadata(self):
        """Load exact M/N/S timestamps and URLs from supplied JSON files.

        Those files may live beside legacy L21--L30 OCR metadata. Only N/M/S
        video IDs are consumed here, and their per-file ``idx`` values are not
        treated as a global embedding index.
        """
        root = self.metadata_dir
        timestamps, fps_by_video, video_urls = {}, {}, {}
        if root is None or not root.exists():
            return timestamps, fps_by_video, video_urls

        paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in paths:
            if not VIDEO_ID_RE.fullmatch(path.stem):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            for record in payload:
                if not isinstance(record, dict):
                    continue
                video_id = str(record.get("video_id") or path.stem).upper()
                if not VIDEO_ID_RE.fullmatch(video_id):
                    continue
                raw_frame = record.get("frame_idx")
                if raw_frame is None:
                    raw_frame = record.get("frame_id")
                raw_time = record.get("pts_time")
                if raw_time is None:
                    raw_time = record.get("frame_stamp")
                try:
                    frame_idx = int(raw_frame)
                    pts_time = float(raw_time)
                except (TypeError, ValueError):
                    continue
                timestamps[(video_id, frame_idx)] = max(0.0, pts_time)
                try:
                    fps = float(record.get("fps"))
                    if fps > 0:
                        fps_by_video[video_id] = fps
                except (TypeError, ValueError):
                    pass
                video_url = str(record.get("video_url") or "").strip()
                if video_url:
                    video_urls[video_id] = video_url
        return timestamps, fps_by_video, video_urls

    def _infer_fps(self, map_fps):
        result = {str(key).upper(): float(value) for key, value in map_fps.items()}
        for video_id, rows in self.rows_by_video.items():
            if video_id in result:
                continue
            frames = np.asarray(sorted({int(self.frame_indices[row]) for row in rows}), dtype=np.int64)
            steps = np.diff(frames)
            steps = steps[steps > 0]
            result[video_id] = float(np.median(steps)) if steps.size else 25.0
        return result

    def _discover_image_assets(self):
        root, assets = self.keyframes_dir, {}
        if not root.exists():
            return assets
        if root.is_file() and root.suffix.casefold() == ".zip":
            zip_paths, image_paths = [root], []
        elif root.is_dir():
            zip_paths, image_paths = [], []
            for current, directories, filenames in os.walk(root):
                # Avoid walking hundreds of thousands of L21-L30 images.
                directories[:] = [
                    name for name in directories
                    if not LEGACY_BATCH_DIR_RE.fullmatch(name)
                ]
                current_path = Path(current)
                for filename in filenames:
                    path = current_path / filename
                    suffix = path.suffix.casefold()
                    if suffix == ".zip":
                        zip_paths.append(path)
                    elif suffix in IMAGE_SUFFIXES:
                        image_paths.append(path)
            zip_paths.sort()
            image_paths.sort()
        else:
            return assets
        for path in image_paths:
            parsed = _video_frame_from_path(path.as_posix())
            if parsed:
                assets.setdefault(parsed, ("file", path, None))
        for zip_path in zip_paths:
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    for member in archive.infolist():
                        if member.is_dir() or Path(member.filename).suffix.casefold() not in IMAGE_SUFFIXES:
                            continue
                        parsed = _video_frame_from_path(member.filename)
                        if parsed:
                            assets.setdefault(parsed, ("zip", zip_path, member.filename))
            except (OSError, zipfile.BadZipFile):
                continue
        return assets

    @property
    def size(self):
        return len(self.relative_paths)

    @property
    def video_count(self):
        return len(self.rows_by_video)

    @property
    def detection_count(self):
        return int(self.detection_available.sum())

    def _semantic_scores(self, query_vector):
        # Query may itself be a read-only row of an mmap during tests/tools.
        vector = np.array(query_vector, dtype=np.float32, copy=True).reshape(-1)
        if vector.shape != (1024,):
            raise ValueError(f"Query vector shape sai: {vector.shape}")
        vector = vector[:self.search_dims]
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Query vector không hợp lệ.")
        vector /= norm
        scores, chunk_size = np.empty(self.size, dtype=np.float32), 4096
        for shard_start, _, embeddings in self.embedding_shards:
            for start in range(0, embeddings.shape[0], chunk_size):
                end = min(start + chunk_size, embeddings.shape[0])
                scores[shard_start + start:shard_start + end] = np.asarray(embeddings[start:end]) @ vector
        return scores

    def search(self, query, query_vector, top_k=100):
        parsed = parse_traffic_query(query)
        scores = self._semantic_scores(query_vector).astype(np.float64)
        eligible = np.ones(self.size, dtype=bool)
        requested_counts = []
        for vehicle in parsed["vehicles"]:
            counts = self.det_counts[vehicle]
            eligible &= (~self.detection_available) | (counts > 0)
            requested_counts.append(counts.astype(np.float64))
        if requested_counts:
            total = np.sum(requested_counts, axis=0)
            detected_hits = eligible & self.detection_available
            scale = max(float(np.quantile(total[detected_hits], 0.95)) if np.any(detected_hits) else 1.0, 1.0)
            scores += 0.10 * np.clip(total / scale, 0.0, 1.0)

        density = np.log1p(self.vehicle_counts.astype(np.float64))
        available_density = density[self.detection_available]
        density /= max(float(available_density.max()) if available_density.size else 1.0, 1.0)
        if parsed["busy"]:
            scores += 0.10 * density
        if parsed["sparse"]:
            scores += 0.10 * (1.0 - density) * self.detection_available

        eligible_rows = np.flatnonzero(eligible)
        if not len(eligible_rows):
            return []
        # Keyword proximity is a reranker, so evaluate it only on a broad
        # semantic pool rather than scanning half a million captions per query.
        pool_limit = min(len(eligible_rows), max(int(top_k) * 50, 5000))
        if pool_limit < len(eligible_rows):
            chosen = np.argpartition(scores[eligible_rows], -pool_limit)[-pool_limit:]
            candidates = eligible_rows[chosen]
        else:
            candidates = eligible_rows

        color_hits = np.zeros(self.size, dtype=np.float32)
        for color in parsed["colors"]:
            terms = COLOR_CAPTION_TERMS[color]
            hits = np.fromiter(
                (any(_contains_phrase(self.caption_lower[row], term) for term in terms)
                 for row in candidates),
                dtype=bool,
                count=len(candidates),
            )
            color_hits[candidates] += hits
        if parsed["colors"]:
            scores[candidates] += 0.12 * color_hits[candidates] / len(parsed["colors"])

        vehicle_caption_hits = np.zeros(len(candidates), dtype=np.float32)
        for vehicle in parsed["vehicles"]:
            terms = tuple(a for a in VEHICLE_ALIASES[vehicle] if a.isascii() and " " not in a)
            vehicle_caption_hits += np.fromiter(
                (any(_contains_phrase(self.caption_lower[row], term) for term in terms)
                 for row in candidates),
                dtype=bool,
                count=len(candidates),
            )
        if parsed["vehicles"]:
            scores[candidates] += 0.05 * vehicle_caption_hits / len(parsed["vehicles"])
        if len(parsed["colors"]) == len(parsed["vehicles"]) == 1:
            color_terms = COLOR_CAPTION_TERMS[parsed["colors"][0]]
            vehicle_terms = tuple(a for a in VEHICLE_ALIASES[parsed["vehicles"][0]] if a.isascii() and " " not in a)
            pair_hits = np.fromiter(
                (_near_pair(self.caption_lower[row], color_terms, vehicle_terms)
                 for row in candidates),
                dtype=bool,
                count=len(candidates),
            )
            scores[candidates] += 0.10 * pair_hits

        limit = min(max(1, int(top_k)), len(candidates))
        if limit < len(candidates):
            candidates = candidates[np.argpartition(scores[candidates], -limit)[-limit:]]
        candidates = candidates[np.argsort(scores[candidates])[::-1]]
        results = []
        for index in candidates:
            matched = ["caption"]
            if self.detection_available[index]:
                matched.extend(f"detection:{name}" for name in parsed["vehicles"] if self.det_counts[name][index] > 0)
            if color_hits[index] > 0:
                matched.extend(f"color:{name}" for name in parsed["colors"])
            counts = {name: int(values[index]) for name, values in self.det_counts.items() if values[index] > 0}
            results.append({
                "path": self.web_paths[index], "videoId": self.video_ids[index],
                "frame_idx": int(self.frame_indices[index]),
                "pts_time": float(self.timestamps[index]), "score": float(scores[index]),
                "caption": self.captions[index], "vehicle_count": int(self.vehicle_counts[index]),
                "vehicle_counts": counts, "matched_by": matched,
                "detection_available": bool(self.detection_available[index]),
                "image_available": bool(self.image_available[index]),
            })
        return results

    def _canonical_web_path(self, image_path):
        path = unquote(urlsplit(str(image_path or "")).path).replace("\\", "/")
        marker, position = self.web_prefix + "/", path.find(self.web_prefix + "/")
        if position < 0:
            return None
        relative = path[position + len(marker):].strip("/")
        return f"{self.web_prefix}/{quote(relative, safe='/._-')}"

    def row_for_path(self, image_path):
        canonical = self._canonical_web_path(image_path)
        return self.row_by_web_path.get(canonical) if canonical else None

    def metadata_for_path(self, image_path):
        index = self.row_for_path(image_path)
        if index is None:
            return None
        return {
            "n": int(self.frame_indices[index]), "frame_idx": int(self.frame_indices[index]),
            "pts_time": float(self.timestamps[index]), "path": self.web_paths[index],
            "video_id": self.video_ids[index], "caption": self.captions[index],
            "vehicle_count": int(self.vehicle_counts[index]),
            "vehicle_counts": {name: int(v[index]) for name, v in self.det_counts.items() if v[index] > 0},
            "seg_road": float(self.seg_road[index]), "seg_person": float(self.seg_person[index]),
            "seg_vehicle_ratio": float(self.seg_vehicle_ratio[index]),
            "detection_available": bool(self.detection_available[index]),
            "image_available": bool(self.image_available[index]),
        }

    def neighbors(self, image_path, radius=15):
        index = self.row_for_path(image_path)
        if index is None:
            return []
        rows = self.rows_by_video[self.video_ids[index]]
        position = rows.index(index)
        return [self.web_paths[row] for row in rows[max(0, position-radius):position+radius+1]]

    def keyframe_map(self, video_id):
        normalized = str(video_id or "").upper()
        rows = self.rows_by_video.get(normalized)
        if not rows:
            return None
        return {
            "fps": self.fps_by_video.get(normalized),
            "times": [float(self.timestamps[row]) for row in rows],
            "data": [[int(self.frame_indices[row]), int(self.frame_indices[row])] for row in rows],
            "paths": [self.web_paths[row] for row in rows],
        }

    def nearest_video_frame(self, video_id, frame_idx=None, pts_time=None):
        rows = self.rows_by_video.get(str(video_id or "").upper())
        if not rows:
            return None
        if pts_time is not None:
            row = min(rows, key=lambda item: abs(float(self.timestamps[item]) - float(pts_time)))
        else:
            row = min(rows, key=lambda item: abs(int(self.frame_indices[item]) - int(frame_idx)))
        return {"frame_idx": int(self.frame_indices[row]), "pts_time": float(self.timestamps[row]), "path": self.web_paths[row]}

    def image_asset_for_path(self, image_path):
        index = self.row_for_path(image_path)
        if index is None or self.assets_by_row[index] is None:
            return None
        kind, source, member = self.assets_by_row[index]
        suffix = Path(member or source).suffix.casefold()
        mimetype = (
            "image/webp" if suffix == ".webp"
            else mimetypes.types_map.get(suffix, "application/octet-stream")
        )
        if kind == "file":
            return {"kind": "file", "path": source, "mimetype": mimetype}
        try:
            with zipfile.ZipFile(source) as archive:
                payload = archive.read(member)
        except (OSError, KeyError, zipfile.BadZipFile):
            return None
        return {"kind": "bytes", "data": payload, "mimetype": mimetype}
