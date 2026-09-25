"""Search portable Jina NumPy packages with Parquet row mappings.

The official packages keep vectors in ``vectors/part-xxxxx.npy`` and a stable
global row mapping in ``mapping.parquet``.  This module memory-maps every NPY
part, validates the row contract, and exposes result metadata without copying
the multi-gigabyte matrices into RAM.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

import numpy as np
import polars as pl


VIDEO_ID_RE = re.compile(r"^[MNS]\d{2,3}[-_]V\d+$", re.IGNORECASE)
VIDEO_ID_IN_PATH_RE = re.compile(r"(?<![A-Z0-9])([MNS]\d{2,3}[-_]V\d+)(?!\d)", re.IGNORECASE)


def video_id_aliases(value: str) -> tuple[str, ...]:
    """Return underscore/dash variants used by the released AIC artifacts."""

    value = str(value or "").strip().upper()
    if not value:
        return ()
    aliases = [value]
    if "_V" in value:
        aliases.append(value.replace("_V", "-V"))
    elif "-V" in value:
        aliases.append(value.replace("-V", "_V"))
    return tuple(dict.fromkeys(aliases))


def normalized_frame_key(video_id: str, frame_idx: int) -> tuple[str, int]:
    """Normalize only for joins; preserve the official ID in API results."""

    aliases = video_id_aliases(video_id)
    normalized = aliases[-1] if aliases and "_V" in aliases[-1] else aliases[0]
    return normalized.replace("-V", "_V"), int(frame_idx)


def _official_video_id(row_video_id: str, zip_member: str) -> str:
    """Prefer the spelling used by the released keyframe/video filenames."""

    match = VIDEO_ID_IN_PATH_RE.search(str(zip_member or ""))
    candidate = match.group(1).upper() if match else str(row_video_id or "").upper()
    if not VIDEO_ID_RE.fullmatch(candidate):
        raise ValueError(f"video_id không hợp lệ trong portable mapping: {candidate!r}")
    return candidate


class PortableJinaImageIndex:
    """Exact cosine search over one or more portable image-vector packages."""

    REQUIRED_COLUMNS = {
        "video_id", "collection", "frame_id", "fps", "timestamp",
        "image_name", "zip_member", "vector_index", "npy_file", "npy_row",
    }

    def __init__(self, root: str | Path, expected_dimension: int = 1024,
                 web_prefix: str = "/batch2-keyframes/image"):
        self.root = Path(root)
        self.d = int(expected_dimension)
        self.web_prefix = "/" + web_prefix.strip("/")
        self.shards: list[tuple[int, np.ndarray]] = []
        self.package_names: list[str] = []

        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        package_dirs = self._discover_packages()
        if not package_dirs:
            raise FileNotFoundError(
                f"Không tìm thấy portable Jina package dưới {self.root}"
            )

        video_ids: list[str] = []
        collections: list[str] = []
        frame_indices: list[int] = []
        timestamps: list[float] = []
        fps_values: list[float] = []
        image_names: list[str] = []
        source_relpaths: list[str] = []
        video_urls: list[str] = []
        package_by_row: list[str] = []
        global_offset = 0

        for package_dir in package_dirs:
            manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
            if int(manifest.get("dimension", -1)) != self.d:
                raise ValueError(f"Sai dimension trong {package_dir / 'manifest.json'}")
            if str(manifest.get("dtype")) != "float32" or not manifest.get("normalized"):
                raise ValueError(f"Portable package chưa phải float32 L2-normalized: {package_dir}")

            mapping_path = package_dir / "mapping.parquet"
            schema_names = set(pl.read_parquet_schema(mapping_path).names())
            missing = sorted(self.REQUIRED_COLUMNS - schema_names)
            if missing:
                raise ValueError(f"Mapping {mapping_path} thiếu cột: {', '.join(missing)}")
            selected = sorted(self.REQUIRED_COLUMNS | ({"video_url", "source_relpath"} & schema_names))
            mapping = pl.read_parquet(mapping_path, columns=selected)
            row_count = mapping.height
            if row_count != int(manifest.get("rows", -1)):
                raise ValueError(
                    f"Manifest/mapping lệch dòng tại {package_dir}: "
                    f"{manifest.get('rows')} != {row_count}"
                )
            vector_indices = mapping.get_column("vector_index").to_numpy()
            if not np.array_equal(vector_indices, np.arange(row_count, dtype=vector_indices.dtype)):
                raise ValueError(f"vector_index không liên tục trong {mapping_path}")

            package_start = global_offset
            part_rows = 0
            vector_paths = sorted((package_dir / "vectors").glob("part-*.npy"))
            if len(vector_paths) != int(manifest.get("parts", -1)):
                raise ValueError(f"Thiếu vector part trong {package_dir}")
            for vector_path in vector_paths:
                vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
                if vectors.ndim != 2 or vectors.shape[1] != self.d or vectors.dtype != np.float32:
                    raise ValueError(
                        f"Vector part không hợp lệ: {vector_path} {vectors.shape} {vectors.dtype}"
                    )
                sample_indices = sorted({0, len(vectors) // 2, len(vectors) - 1})
                sample = np.asarray(vectors[sample_indices], dtype=np.float32)
                if not np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=2e-3):
                    raise ValueError(f"Vector part chưa L2-normalize: {vector_path}")
                self.shards.append((global_offset + part_rows, vectors))
                part_rows += len(vectors)
            if part_rows != row_count:
                raise ValueError(
                    f"Vector/mapping lệch dòng tại {package_dir}: {part_rows} != {row_count}"
                )

            row_video_ids = mapping.get_column("video_id").to_list()
            zip_members = mapping.get_column("zip_member").fill_null("").to_list()
            official_ids = [
                _official_video_id(video_id, zip_member)
                for video_id, zip_member in zip(row_video_ids, zip_members)
            ]
            row_frames = [int(value) for value in mapping.get_column("frame_id").to_list()]
            row_fps = [float(value or 0.0) for value in mapping.get_column("fps").to_list()]
            row_times = mapping.get_column("timestamp").to_list()
            row_times = [
                max(0.0, float(value))
                if value is not None and math.isfinite(float(value))
                else (float(frame) / fps if fps > 0 else 0.0)
                for value, frame, fps in zip(row_times, row_frames, row_fps)
            ]
            names = [str(value) for value in mapping.get_column("image_name").to_list()]
            relpaths = (
                [str(value or "") for value in mapping.get_column("source_relpath").to_list()]
                if "source_relpath" in mapping.columns else names
            )
            urls = (
                [str(value or "") for value in mapping.get_column("video_url").to_list()]
                if "video_url" in mapping.columns else [""] * row_count
            )

            video_ids.extend(official_ids)
            collections.extend(str(value).upper() for value in mapping.get_column("collection").to_list())
            frame_indices.extend(row_frames)
            timestamps.extend(row_times)
            fps_values.extend(row_fps)
            image_names.extend(names)
            source_relpaths.extend(relpaths)
            video_urls.extend(urls)
            package_by_row.extend([package_dir.name] * row_count)
            self.package_names.append(package_dir.name)
            global_offset = package_start + row_count

        self.video_ids = video_ids
        self.collections = collections
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.fps_values = np.asarray(fps_values, dtype=np.float32)
        self.image_names = image_names
        self.source_relpaths = source_relpaths
        self.video_urls = video_urls
        self.package_by_row = package_by_row
        self.ntotal = len(video_ids)

        self.rows_by_video: dict[str, list[int]] = {}
        self.canonical_video_id: dict[str, str] = {}
        self.row_by_frame: dict[tuple[str, int], int] = {}
        self.video_url_by_id: dict[str, str] = {}
        for row, (video_id, frame_idx, video_url) in enumerate(
            zip(self.video_ids, self.frame_indices, self.video_urls)
        ):
            self.rows_by_video.setdefault(video_id, []).append(row)
            for alias in video_id_aliases(video_id):
                self.canonical_video_id[alias] = video_id
                key = normalized_frame_key(alias, int(frame_idx))
                if key in self.row_by_frame and self.row_by_frame[key] != row:
                    raise ValueError(f"Trùng portable image mapping: {key}")
                self.row_by_frame[key] = row
                if video_url:
                    self.video_url_by_id[alias] = video_url
        for rows in self.rows_by_video.values():
            rows.sort(key=lambda row: (self.timestamps[row], self.frame_indices[row]))

    def _discover_packages(self) -> list[Path]:
        packages = []
        for manifest_path in self.root.rglob("manifest.json"):
            package_dir = manifest_path.parent
            if (package_dir / "mapping.parquet").is_file() and (package_dir / "vectors").is_dir():
                packages.append(package_dir)
        return sorted(set(packages))

    @property
    def video_count(self) -> int:
        return len(self.rows_by_video)

    def _video_rows(self, video_id: str) -> list[int] | None:
        canonical = self.canonical_video_id.get(str(video_id or "").upper())
        return self.rows_by_video.get(canonical) if canonical else None

    def path_for_row(self, row: int) -> str:
        return (
            f"{self.web_prefix}/{quote(self.video_ids[row], safe='-_')}/"
            f"{quote(self.image_names[row], safe='._-')}"
        )

    def row_for_path(self, image_path: str) -> int | None:
        path = unquote(urlsplit(str(image_path or "")).path).replace("\\", "/")
        marker = self.web_prefix + "/"
        position = path.find(marker)
        if position < 0:
            return None
        parts = PurePosixPath(path[position + len(marker):]).parts
        if len(parts) < 2 or not Path(parts[-1]).stem.isdigit():
            return None
        return self.row_by_frame.get(normalized_frame_key(parts[-2], int(Path(parts[-1]).stem)))

    def _result(self, row: int, score: float) -> dict:
        collection = self.collections[row]
        return {
            "path": self.path_for_row(row),
            "videoId": self.video_ids[row],
            "frame_idx": int(self.frame_indices[row]),
            "pts_time": float(self.timestamps[row]),
            "score": float(score),
            "batch": "final" if collection.startswith("S") else "batch2",
            "collection": collection,
            "matched_by": ["image"],
            # The portable package contains embeddings + row mappings, not
            # necessarily the original keyframe files.  app.py overlays the
            # real availability from the shared keyframe asset index.
            "image_available": False,
        }

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict]:
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.shape != (self.d,):
            raise ValueError(f"Query vector shape sai: {vector.shape}, cần ({self.d},)")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Query vector không hợp lệ.")
        vector = vector / norm
        top_k = max(1, min(int(top_k), self.ntotal))

        candidate_rows: list[np.ndarray] = []
        candidate_scores: list[np.ndarray] = []
        for offset, vectors in self.shards:
            scores = np.asarray(vectors @ vector, dtype=np.float32)
            local_k = min(top_k, len(scores))
            if local_k < len(scores):
                local_rows = np.argpartition(scores, -local_k)[-local_k:]
            else:
                local_rows = np.arange(len(scores))
            candidate_rows.append(local_rows.astype(np.int64, copy=False) + offset)
            candidate_scores.append(scores[local_rows])

        rows = np.concatenate(candidate_rows)
        scores = np.concatenate(candidate_scores)
        if len(rows) > top_k:
            chosen = np.argpartition(scores, -top_k)[-top_k:]
            rows, scores = rows[chosen], scores[chosen]
        order = np.argsort(scores)[::-1]
        return [self._result(int(rows[i]), float(scores[i])) for i in order]

    def metadata_for_path(self, image_path: str) -> dict | None:
        row = self.row_for_path(image_path)
        if row is None:
            return None
        return {
            "n": int(self.frame_indices[row]),
            "frame_idx": int(self.frame_indices[row]),
            "pts_time": float(self.timestamps[row]),
            "fps": float(self.fps_values[row]) if self.fps_values[row] > 0 else None,
            "path": self.path_for_row(row),
            "video_id": self.video_ids[row],
            "collection": self.collections[row],
            "source_relpath": self.source_relpaths[row],
            "image_available": False,
        }

    def neighbors(self, image_path: str, radius: int = 15) -> list[str]:
        row = self.row_for_path(image_path)
        if row is None:
            return []
        rows = self.rows_by_video[self.video_ids[row]]
        position = rows.index(row)
        return [self.path_for_row(item) for item in rows[max(0, position-radius):position+radius+1]]

    def keyframe_map(self, video_id: str) -> dict | None:
        rows = self._video_rows(video_id)
        if not rows:
            return None
        fps = next((float(self.fps_values[row]) for row in rows if self.fps_values[row] > 0), None)
        return {
            "fps": fps,
            "times": [float(self.timestamps[row]) for row in rows],
            "data": [[int(self.frame_indices[row]), int(self.frame_indices[row])] for row in rows],
            "paths": [self.path_for_row(row) for row in rows],
        }

    def results_for_video(self, video_id: str, limit: int | None = None) -> list[dict]:
        rows = self._video_rows(video_id) or []
        if limit is not None:
            rows = rows[:max(0, int(limit))]
        return [self._result(row, float(self.timestamps[row])) for row in rows]

    def results_around_time(self, video_id: str, target_time: float,
                            time_border: float, limit: int) -> list[dict]:
        rows = self._video_rows(video_id) or []
        rows = [
            row for row in rows
            if abs(float(self.timestamps[row]) - float(target_time)) <= float(time_border)
        ]
        rows.sort(key=lambda row: (
            abs(float(self.timestamps[row]) - float(target_time)),
            float(self.timestamps[row]),
        ))
        return [self._result(row, 0.0) for row in rows[:max(0, int(limit))]]

    def nearest_video_frame(self, video_id: str, frame_idx=None, pts_time=None) -> dict | None:
        rows = self._video_rows(video_id)
        if not rows:
            return None
        if pts_time is not None:
            row = min(rows, key=lambda item: abs(float(self.timestamps[item]) - float(pts_time)))
        else:
            row = min(rows, key=lambda item: abs(int(self.frame_indices[item]) - int(frame_idx)))
        return {
            "frame_idx": int(self.frame_indices[row]),
            "pts_time": float(self.timestamps[row]),
            "path": self.path_for_row(row),
        }
