"""Traffic retrieval for the Batch 2 N081-N100 demo.

Caption embeddings provide semantic/color evidence. Detection columns provide
hard vehicle constraints and count evidence. Timestamps come from the canonical
map-keyframes CSVs because the detector export's timestamp_s is not reliable.
"""

from __future__ import annotations

import csv
import math
import re
import unicodedata
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
    "black": ("den", "black"),
    "white": ("trang", "white"),
    "red": ("do", "red"),
    "blue": ("xanh duong", "xanh da troi", "blue", "teal", "navy"),
    "green": ("xanh la", "green"),
    "yellow_gold": ("yellow", "gold", "golden"),
    "orange": ("cam", "orange"),
    "gray_silver": ("xam", "ghi", "bac", "gray", "grey", "silver"),
    "brown": ("nau", "brown", "beige"),
    "pink": ("hong", "pink"),
    "purple": ("purple", "violet"),
}

COLOR_CAPTION_TERMS = {
    name: tuple(alias for alias in aliases if alias.isascii() and " " not in alias)
    for name, aliases in COLOR_ALIASES.items()
}

BUSY_TERMS = ("dong xe", "nhieu xe", "ket xe", "un tac", "busy", "crowded", "congestion", "heavy traffic")
SPARSE_TERMS = ("vang xe", "it xe", "duong vang", "empty road", "light traffic", "few vehicles")

DETECTION_COLUMNS = {
    "car": "det_car_count",
    "motorcycle": "det_motorcycle_count",
    "bus": "det_bus_count",
    "truck": "det_truck_count",
    "bicycle": "det_bicycle_count",
}


def normalize_text(value: str) -> str:
    value = str(value or "").casefold().replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def parse_traffic_query(query: str) -> dict:
    normalized = normalize_text(query)
    vehicles = [
        name for name, aliases in VEHICLE_ALIASES.items()
        if any(_contains_phrase(normalized, alias) for alias in aliases)
    ]
    colors = [
        name for name, aliases in COLOR_ALIASES.items()
        if any(_contains_phrase(normalized, alias) for alias in aliases)
    ]
    # After accent removal, "tìm" collides with "tím" and "vắng" with
    # "vàng". Require an explicit color phrase for these two Vietnamese words.
    if _contains_phrase(normalized, "mau tim") and "purple" not in colors:
        colors.append("purple")
    if _contains_phrase(normalized, "mau vang") and "yellow_gold" not in colors:
        colors.append("yellow_gold")
    # Bare "xanh" is commonly used for blue in short vehicle queries. Do not
    # add it when the query already says xanh lá/xanh dương explicitly.
    if (
        _contains_phrase(normalized, "xanh")
        and "blue" not in colors
        and "green" not in colors
    ):
        colors.append("blue")
    return {
        "normalized": normalized,
        "vehicles": vehicles,
        "colors": colors,
        "busy": any(_contains_phrase(normalized, term) for term in BUSY_TERMS),
        "sparse": any(_contains_phrase(normalized, term) for term in SPARSE_TERMS),
    }


def expanded_query(query: str, parsed: dict) -> str:
    english = []
    english.extend(color.replace("_", " ") for color in parsed["colors"])
    english.extend(parsed["vehicles"])
    if parsed["busy"]:
        english.extend(("busy", "heavy traffic"))
    if parsed["sparse"]:
        english.extend(("empty road", "few vehicles"))
    return f"{query}. Traffic scene: {' '.join(english)}" if english else query


def _near_pair(caption: str, color_terms: tuple[str, ...], vehicle_terms: tuple[str, ...]) -> bool:
    words = re.findall(r"[a-z0-9]+", caption.casefold())
    color_positions = [i for i, word in enumerate(words) if word in color_terms]
    vehicle_positions = [i for i, word in enumerate(words) if word in vehicle_terms]
    return any(abs(left - right) <= 4 for left in color_positions for right in vehicle_positions)


class TrafficSearchIndex:
    """Aligned Batch 2 caption, detection, keyframe-map and image index."""

    REQUIRED_DETECTION_COLUMNS = (
        "member_path",
        "video_id",
        "frame_idx",
        "det_person_count",
        "det_traffic_light_count",
        "det_car_count",
        "det_motorcycle_count",
        "det_bus_count",
        "det_truck_count",
        "det_bicycle_count",
        "vehicle_count",
        "seg_road",
        "seg_person",
        "seg_vehicle_ratio",
    )

    def __init__(
        self,
        caption_dir: str | Path,
        detection_path: str | Path,
        keyframes_dir: str | Path,
        map_dir: str | Path,
        web_prefix: str = "/batch2-keyframes",
    ):
        self.caption_dir = Path(caption_dir)
        self.detection_path = Path(detection_path)
        self.keyframes_dir = Path(keyframes_dir)
        self.map_dir = Path(map_dir)
        self.web_prefix = "/" + web_prefix.strip("/")

        mapping_path = self.caption_dir / "caption_mapping.csv"
        embedding_path = self.caption_dir / "caption_embeddings.npy"
        for path in (mapping_path, embedding_path, self.detection_path, self.keyframes_dir, self.map_dir):
            if not path.exists():
                raise FileNotFoundError(path)

        self.embeddings = np.load(embedding_path, mmap_mode="r")
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 1024:
            raise ValueError(f"Caption embedding shape không hợp lệ: {self.embeddings.shape}")

        with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
            mapping_rows = list(csv.DictReader(stream))
        if len(mapping_rows) != self.embeddings.shape[0]:
            raise ValueError(
                f"Caption mapping/embedding lệch dòng: {len(mapping_rows)} != {self.embeddings.shape[0]}"
            )

        detector = pl.read_parquet(
            self.detection_path,
            columns=list(self.REQUIRED_DETECTION_COLUMNS),
        )
        detector_by_path = {
            row["member_path"]: row for row in detector.iter_rows(named=True)
        }
        if len(detector_by_path) != detector.height:
            raise ValueError("Detection member_path bị trùng.")

        timestamps, fps_by_video = self._load_keyframe_maps()
        count = len(mapping_rows)
        self.relative_paths = []
        self.web_paths = []
        self.video_ids = []
        self.frame_indices = np.empty(count, dtype=np.int64)
        self.timestamps = np.empty(count, dtype=np.float64)
        self.captions = []
        self.caption_lower = []
        self.vehicle_counts = np.empty(count, dtype=np.int32)
        self.person_counts = np.empty(count, dtype=np.int32)
        self.traffic_light_counts = np.empty(count, dtype=np.int32)
        self.seg_road = np.empty(count, dtype=np.float32)
        self.seg_person = np.empty(count, dtype=np.float32)
        self.seg_vehicle_ratio = np.empty(count, dtype=np.float32)
        self.det_counts = {
            name: np.empty(count, dtype=np.int16) for name in DETECTION_COLUMNS
        }

        missing_detection = []
        missing_time = []
        for index, mapping in enumerate(mapping_rows):
            relative_path = PurePosixPath(mapping["relative_path"].replace("\\", "/")).as_posix()
            detection = detector_by_path.get(relative_path)
            if detection is None:
                missing_detection.append(relative_path)
                continue
            video_id = str(detection["video_id"])
            frame_idx = int(detection["frame_idx"])
            timestamp = timestamps.get((video_id, frame_idx))
            if timestamp is None:
                missing_time.append((video_id, frame_idx))
                continue

            try:
                image_relative = PurePosixPath(relative_path).relative_to("keyframes").as_posix()
            except ValueError as exc:
                raise ValueError(f"Caption path không nằm dưới keyframes/: {relative_path}") from exc
            image_path = self.keyframes_dir.joinpath(*PurePosixPath(image_relative).parts)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            caption = str(mapping.get("caption") or "").strip()
            self.relative_paths.append(relative_path)
            self.web_paths.append(f"{self.web_prefix}/{quote(image_relative, safe='/')}")
            self.video_ids.append(video_id)
            self.frame_indices[index] = frame_idx
            self.timestamps[index] = timestamp
            self.captions.append(caption)
            self.caption_lower.append(caption.casefold())
            self.vehicle_counts[index] = int(detection["vehicle_count"])
            self.person_counts[index] = int(detection["det_person_count"])
            self.traffic_light_counts[index] = int(detection["det_traffic_light_count"])
            self.seg_road[index] = float(detection["seg_road"])
            self.seg_person[index] = float(detection["seg_person"])
            self.seg_vehicle_ratio[index] = float(detection["seg_vehicle_ratio"])
            for name, column in DETECTION_COLUMNS.items():
                self.det_counts[name][index] = int(detection[column])

        if missing_detection or missing_time:
            raise ValueError(
                f"Không align được Batch 2: missing_detection={len(missing_detection)}, "
                f"missing_timestamp={len(missing_time)}"
            )
        if len(self.relative_paths) != count:
            raise ValueError("Traffic index chưa điền đủ mọi caption row.")

        self.fps_by_video = fps_by_video
        self.row_by_web_path = {path: index for index, path in enumerate(self.web_paths)}
        self.rows_by_video = {}
        for index, video_id in enumerate(self.video_ids):
            self.rows_by_video.setdefault(video_id, []).append(index)
        for video_id, indices in self.rows_by_video.items():
            indices.sort(key=lambda row: (self.timestamps[row], self.frame_indices[row]))

    def _load_keyframe_maps(self):
        timestamps = {}
        fps_by_video = {}
        for path in sorted(self.map_dir.glob("*.csv")):
            video_id = path.stem
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                continue
            fps_values = {round(float(row["fps"]), 6) for row in rows}
            if len(fps_values) != 1:
                raise ValueError(f"FPS không nhất quán: {path}")
            fps_by_video[video_id] = float(rows[0]["fps"])
            for row in rows:
                key = (video_id, int(row["frame_id"]))
                if key in timestamps:
                    raise ValueError(f"Trùng map-keyframe: {key}")
                timestamps[key] = float(row["timestamp"])
        return timestamps, fps_by_video

    @property
    def size(self):
        return len(self.relative_paths)

    @property
    def video_count(self):
        return len(self.rows_by_video)

    def _semantic_scores(self, query_vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape != (self.embeddings.shape[1],):
            raise ValueError(f"Query vector shape sai: {vector.shape}")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Query vector không hợp lệ.")
        vector = vector / norm
        scores = np.empty(self.size, dtype=np.float32)
        chunk_size = 4096
        for start in range(0, self.size, chunk_size):
            end = min(start + chunk_size, self.size)
            scores[start:end] = np.asarray(self.embeddings[start:end]) @ vector
        return scores

    def search(self, query: str, query_vector: np.ndarray, top_k: int = 100) -> list[dict]:
        parsed = parse_traffic_query(query)
        scores = self._semantic_scores(query_vector).astype(np.float64)
        eligible = np.ones(self.size, dtype=bool)
        matched_labels = [[] for _ in range(self.size)]

        requested_counts = []
        for vehicle in parsed["vehicles"]:
            counts = self.det_counts[vehicle]
            eligible &= counts > 0
            requested_counts.append(counts.astype(np.float64))
        if requested_counts:
            total = np.sum(requested_counts, axis=0)
            scale = max(float(np.quantile(total[eligible], 0.95)) if np.any(eligible) else 1.0, 1.0)
            scores += 0.10 * np.clip(total / scale, 0.0, 1.0)

        color_hits = np.zeros(self.size, dtype=np.float64)
        for color in parsed["colors"]:
            terms = COLOR_CAPTION_TERMS[color]
            hits = np.fromiter(
                (any(_contains_phrase(caption, term) for term in terms) for caption in self.caption_lower),
                dtype=bool,
                count=self.size,
            )
            color_hits += hits
        if parsed["colors"]:
            scores += 0.12 * color_hits / len(parsed["colors"])

        vehicle_caption_hits = np.zeros(self.size, dtype=np.float64)
        for vehicle in parsed["vehicles"]:
            terms = tuple(alias for alias in VEHICLE_ALIASES[vehicle] if alias.isascii() and " " not in alias)
            hits = np.fromiter(
                (any(_contains_phrase(caption, term) for term in terms) for caption in self.caption_lower),
                dtype=bool,
                count=self.size,
            )
            vehicle_caption_hits += hits
        if parsed["vehicles"]:
            scores += 0.05 * vehicle_caption_hits / len(parsed["vehicles"])

        if len(parsed["colors"]) == 1 and len(parsed["vehicles"]) == 1:
            color_terms = COLOR_CAPTION_TERMS[parsed["colors"][0]]
            vehicle_terms = tuple(
                alias for alias in VEHICLE_ALIASES[parsed["vehicles"][0]]
                if alias.isascii() and " " not in alias
            )
            pair_hits = np.fromiter(
                (_near_pair(caption, color_terms, vehicle_terms) for caption in self.caption_lower),
                dtype=bool,
                count=self.size,
            )
            scores += 0.10 * pair_hits

        density = np.log1p(self.vehicle_counts.astype(np.float64))
        density /= max(float(density.max()), 1.0)
        if parsed["busy"]:
            scores += 0.10 * density
        if parsed["sparse"]:
            scores += 0.10 * (1.0 - density)

        candidate_indices = np.flatnonzero(eligible)
        if not len(candidate_indices):
            return []
        limit = min(max(1, int(top_k)), len(candidate_indices))
        candidate_scores = scores[candidate_indices]
        if limit < len(candidate_indices):
            selected = np.argpartition(candidate_scores, -limit)[-limit:]
            candidate_indices = candidate_indices[selected]
        candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        results = []
        for index in candidate_indices:
            matched = ["caption"]
            matched.extend(f"detection:{name}" for name in parsed["vehicles"])
            if color_hits[index] > 0:
                matched.extend(f"color:{name}" for name in parsed["colors"])
            counts = {
                name: int(values[index])
                for name, values in self.det_counts.items()
                if values[index] > 0
            }
            results.append({
                "path": self.web_paths[index],
                "videoId": self.video_ids[index],
                "frame_idx": int(self.frame_indices[index]),
                "pts_time": float(self.timestamps[index]),
                "score": float(scores[index]),
                "caption": self.captions[index],
                "vehicle_count": int(self.vehicle_counts[index]),
                "vehicle_counts": counts,
                "matched_by": matched,
            })
        return results

    def _canonical_web_path(self, image_path: str) -> str | None:
        parsed = urlsplit(str(image_path or ""))
        path = unquote(parsed.path).replace("\\", "/")
        marker = self.web_prefix + "/"
        position = path.find(marker)
        if position < 0:
            return None
        relative = path[position + len(marker):].strip("/")
        return f"{self.web_prefix}/{quote(relative, safe='/')}"

    def row_for_path(self, image_path: str) -> int | None:
        canonical = self._canonical_web_path(image_path)
        return self.row_by_web_path.get(canonical) if canonical else None

    def metadata_for_path(self, image_path: str) -> dict | None:
        index = self.row_for_path(image_path)
        if index is None:
            return None
        return {
            "n": int(self.frame_indices[index]),
            "frame_idx": int(self.frame_indices[index]),
            "pts_time": float(self.timestamps[index]),
            "path": self.web_paths[index],
            "video_id": self.video_ids[index],
            "caption": self.captions[index],
            "vehicle_count": int(self.vehicle_counts[index]),
            "vehicle_counts": {
                name: int(values[index])
                for name, values in self.det_counts.items()
                if values[index] > 0
            },
            "seg_road": float(self.seg_road[index]),
            "seg_person": float(self.seg_person[index]),
            "seg_vehicle_ratio": float(self.seg_vehicle_ratio[index]),
        }

    def neighbors(self, image_path: str, radius: int = 15) -> list[str]:
        index = self.row_for_path(image_path)
        if index is None:
            return []
        rows = self.rows_by_video[self.video_ids[index]]
        position = rows.index(index)
        start = max(0, position - radius)
        end = min(len(rows), position + radius + 1)
        return [self.web_paths[row] for row in rows[start:end]]

    def keyframe_map(self, video_id: str) -> dict | None:
        rows = self.rows_by_video.get(str(video_id or "").upper())
        if not rows:
            return None
        return {
            "fps": self.fps_by_video.get(str(video_id).upper()),
            "times": [float(self.timestamps[row]) for row in rows],
            "data": [
                [int(self.frame_indices[row]), int(self.frame_indices[row])]
                for row in rows
            ],
            "paths": [self.web_paths[row] for row in rows],
        }

    def nearest_video_frame(self, video_id: str, frame_idx: int | None = None, pts_time: float | None = None):
        rows = self.rows_by_video.get(str(video_id or "").upper())
        if not rows:
            return None
        if pts_time is not None:
            row = min(rows, key=lambda item: abs(float(self.timestamps[item]) - float(pts_time)))
        else:
            row = min(rows, key=lambda item: abs(int(self.frame_indices[item]) - int(frame_idx)))
        return {
            "frame_idx": int(self.frame_indices[row]),
            "pts_time": float(self.timestamps[row]),
            "path": self.web_paths[row],
        }
