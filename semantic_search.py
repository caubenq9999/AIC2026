"""Jina-only encoders and exact search over normalized NPY shards."""

from __future__ import annotations

import gc
import importlib.util
import io
import json
import os
import threading
from pathlib import Path
import zipfile

import numpy as np
import torch


class ModelUnavailableError(RuntimeError):
    """Raised when a lazy model cannot be loaded for inference."""


def normalize_vectors(vectors, expected_dimension: int) -> np.ndarray:
    """Return a finite, contiguous, row-wise L2-normalized float32 matrix."""
    matrix = np.array(vectors, dtype=np.float32, copy=True, order="C")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"Vector phải có shape (n, d), nhận được {matrix.shape}.")
    if matrix.shape[1] != expected_dimension:
        raise ValueError(
            f"Vector sai số chiều: nhận {matrix.shape[1]}, cần {expected_dimension}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Vector chứa NaN hoặc infinity.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Không thể chuẩn hóa vector có norm bằng 0.")
    matrix /= norms
    return matrix


def normalize_query_vector(vector, expected_dimension: int) -> np.ndarray:
    matrix = normalize_vectors(vector, expected_dimension)
    if matrix.shape[0] != 1:
        raise ValueError(f"Query vector phải có shape (1, d), nhận được {matrix.shape}.")
    return matrix


class ShardedNpyIndex:
    """Exact inner-product search over normalized collection-level NPY shards."""

    def __init__(
        self,
        name: str,
        vectors_dir: str | Path,
        image_records: list[dict],
        expected_dimension: int,
    ):
        self.name = name
        self.vectors_dir = Path(vectors_dir)
        self.d = expected_dimension
        self.shards: list[tuple[int, np.ndarray]] = []

        if not self.vectors_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy vector {name}: {self.vectors_dir}")

        shard_paths = sorted(
            (
                path
                for path in self.vectors_dir.glob("L*.npy")
                if path.stem.startswith("L") and path.stem[1:].isdigit()
            ),
            key=lambda path: int(path.stem[1:]),
        )
        if not shard_paths:
            raise FileNotFoundError(f"Không có shard L*.npy trong {self.vectors_dir}")

        offset = 0
        for shard_path in shard_paths:
            collection = shard_path.stem.upper()
            vectors = np.load(shard_path, mmap_mode="r")
            if vectors.ndim != 2 or vectors.shape[1] != expected_dimension:
                raise ValueError(
                    f"Shard {shard_path.name} có shape {vectors.shape}, "
                    f"cần (N, {expected_dimension})."
                )
            if vectors.dtype != np.float32:
                raise ValueError(
                    f"Shard {shard_path.name} có dtype {vectors.dtype}, cần float32."
                )

            end = offset + len(vectors)
            if end > len(image_records):
                raise ValueError(f"Vector {name} có nhiều dòng hơn metadata.")
            record_collections = {
                str(record.get("video_id", "")).split("_", 1)[0].upper()
                for record in image_records[offset:end]
            }
            if record_collections != {collection}:
                raise ValueError(
                    f"Thứ tự shard {name}/{collection} không khớp metadata tại "
                    f"offset {offset}:{end}: {sorted(record_collections)}"
                )

            sample_indices = sorted({0, len(vectors) // 2, len(vectors) - 1})
            sample = np.asarray(vectors[sample_indices], dtype=np.float32)
            norms = np.linalg.norm(sample, axis=1)
            if not np.allclose(norms, 1.0, atol=2e-3):
                raise ValueError(
                    f"Vector {name}/{collection} chưa L2-normalize; norms={norms.tolist()}"
                )

            self.shards.append((offset, vectors))
            offset = end

        if offset != len(image_records):
            raise ValueError(
                f"Số vector {name} không khớp metadata: vectors={offset}, "
                f"metadata={len(image_records)}"
            )
        self.ntotal = offset

    def search(self, query_vector: np.ndarray, top_k: int):
        query_vector = normalize_query_vector(query_vector, self.d)
        top_k = max(1, min(int(top_k), self.ntotal))

        shard_distances = []
        shard_indices = []
        for offset, vectors in self.shards:
            local_k = min(top_k, len(vectors))
            scores = np.asarray(vectors @ query_vector[0], dtype=np.float32)
            if local_k < len(scores):
                local_indices = np.argpartition(scores, -local_k)[-local_k:]
                local_indices = local_indices[np.argsort(scores[local_indices])[::-1]]
            else:
                local_indices = np.argsort(scores)[::-1]
            shard_distances.append(scores[local_indices])
            shard_indices.append(local_indices.astype(np.int64, copy=False) + offset)

        all_distances = np.concatenate(shard_distances)
        all_indices = np.concatenate(shard_indices)
        if len(all_distances) > top_k:
            candidate_positions = np.argpartition(all_distances, -top_k)[-top_k:]
            candidate_positions = candidate_positions[
                np.argsort(all_distances[candidate_positions])[::-1]
            ]
        else:
            candidate_positions = np.argsort(all_distances)[::-1]

        return (
            all_distances[candidate_positions].reshape(1, -1),
            all_indices[candidate_positions].reshape(1, -1),
        )


def prepare_npz_archive_cache(
    archive_dir: str | Path,
    cache_dir: str | Path,
    image_records: list[dict],
    expected_dimension: int,
) -> Path:
    """Convert Apple-CLIP NPZ exports into collection-level mmap NPY files.

    Each collection may be a downloaded ``Lxx-*.zip`` or an extracted
    ``Lxx/shard_*.npz`` folder. The encoder export stores 1,000 rows per NPZ and
    includes the image name for every vector. Validate those names against
    canonical metadata while writing the cache so a stale or shuffled embedding
    can never silently point at the wrong keyframe.
    """

    archive_dir = Path(archive_dir)
    cache_dir = Path(cache_dir)
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy Apple-CLIP artifacts: {archive_dir}")

    source_paths: dict[str, Path] = {}
    for path in archive_dir.glob("L*.zip"):
        collection = path.name.split("-", 1)[0].upper()
        if collection[1:].isdigit():
            if collection in source_paths:
                raise ValueError(f"Trùng archive Apple-CLIP cho {collection}.")
            source_paths[collection] = path
    # Nếu tồn tại cả ZIP và folder đã giải nén, ưu tiên folder để khỏi giải nén
    # từng NPZ vào RAM ở mỗi lần dựng lại cache.
    for path in archive_dir.iterdir():
        collection = path.name.upper()
        if path.is_dir() and collection.startswith("L") and collection[1:].isdigit():
            source_paths[collection] = path

    expected_by_collection: dict[str, list[str]] = {}
    for record in image_records:
        video_id = str(record.get("video_id", "")).upper()
        collection = video_id.split("_", 1)[0]
        frame_name = Path(str(record.get("path", "")).replace("\\", "/")).name
        expected_by_collection.setdefault(collection, []).append(
            f"{video_id}/{frame_name}"
        )

    missing = sorted(set(expected_by_collection) - set(source_paths))
    if missing:
        raise FileNotFoundError(
            "Apple-CLIP artifacts chưa đầy đủ; thiếu " + ", ".join(missing)
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    for collection in sorted(expected_by_collection, key=lambda value: int(value[1:])):
        source_path = source_paths[collection]
        expected_names = expected_by_collection[collection]
        output_path = cache_dir / f"{collection}.npy"
        marker_path = cache_dir / f"{collection}.ready.json"
        extracted_shards = []
        if source_path.is_dir():
            extracted_shards = sorted(
                source_path.rglob("shard_*.npz"),
                key=lambda path: int(path.stem.rsplit("_", 1)[1]),
            )
            source_signature = {
                "folder": source_path.name,
                "shards": [
                    [str(path.relative_to(source_path)).replace("\\", "/"), path.stat().st_size]
                    for path in extracted_shards
                ],
                "rows": len(expected_names),
                "dimension": expected_dimension,
            }
        else:
            # Giữ schema marker cũ để máy đã dựng cache từ ZIP không phải làm lại.
            source_signature = {
                "archive": source_path.name,
                "archive_size": source_path.stat().st_size,
                "rows": len(expected_names),
                "dimension": expected_dimension,
            }

        cache_is_current = False
        if output_path.is_file() and marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                cached = np.load(output_path, mmap_mode="r")
                cache_is_current = (
                    marker == source_signature
                    and cached.shape == (len(expected_names), expected_dimension)
                    and cached.dtype == np.float32
                )
                del cached
            except (OSError, ValueError, json.JSONDecodeError):
                cache_is_current = False
        if cache_is_current:
            continue

        temp_output = output_path.with_suffix(".tmp.npy")
        temp_marker = marker_path.with_suffix(".tmp.json")
        temp_output.unlink(missing_ok=True)
        temp_marker.unlink(missing_ok=True)
        destination = np.lib.format.open_memmap(
            temp_output,
            mode="w+",
            dtype=np.float32,
            shape=(len(expected_names), expected_dimension),
        )
        offset = 0

        def validate_shard_order(shard_items):
            actual_shards = [
                int(Path(item).stem.rsplit("_", 1)[1]) for item in shard_items
            ]
            if not shard_items or actual_shards != list(range(len(shard_items))):
                raise ValueError(
                    f"Shard Apple-CLIP {collection} bị thiếu hoặc không liên tục."
                )

        def append_shard(shard, shard_name):
            nonlocal offset
            vectors = shard["embeddings"]
            image_names = [str(value) for value in shard["image_names"]]
            if vectors.ndim != 2 or vectors.shape[1] != expected_dimension:
                raise ValueError(
                    f"{shard_name} có shape {vectors.shape}, "
                    f"cần (N, {expected_dimension})."
                )
            end = offset + len(vectors)
            if image_names != expected_names[offset:end]:
                raise ValueError(
                    f"Thứ tự image_names sai tại {collection} "
                    f"offset {offset}:{end}."
                )
            errors = json.loads(str(shard["errors_json"][0]))
            if errors:
                raise ValueError(f"{shard_name} chứa {len(errors)} lỗi encode.")
            destination[offset:end] = np.asarray(vectors, dtype=np.float32)
            offset = end

        try:
            if source_path.is_dir():
                validate_shard_order(extracted_shards)
                for shard_path in extracted_shards:
                    with np.load(shard_path, allow_pickle=False) as shard:
                        append_shard(shard, str(shard_path.relative_to(source_path)))
            else:
                with zipfile.ZipFile(source_path) as archive:
                    shard_names = sorted(
                        (
                            name
                            for name in archive.namelist()
                            if Path(name).name.startswith("shard_")
                            and name.lower().endswith(".npz")
                        ),
                        key=lambda name: int(Path(name).stem.rsplit("_", 1)[1]),
                    )
                    validate_shard_order(shard_names)
                    for shard_name in shard_names:
                        with np.load(
                            io.BytesIO(archive.read(shard_name)), allow_pickle=False
                        ) as shard:
                            append_shard(shard, shard_name)

            if offset != len(expected_names):
                raise ValueError(
                    f"Số Apple-CLIP vector {collection} không khớp metadata: "
                    f"vectors={offset}, metadata={len(expected_names)}."
                )
            destination.flush()
            destination = None
            temp_marker.write_text(
                json.dumps(source_signature, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_output, output_path)
            os.replace(temp_marker, marker_path)
        except Exception:
            destination = None
            temp_output.unlink(missing_ok=True)
            temp_marker.unlink(missing_ok=True)
            raise

    return cache_dir


class AppleClipTextEncoder:
    """Lazy OpenCLIP text encoder matching the finetuned Apple image space."""

    MODEL_NAME = "ViT-H-14-378-quickgelu"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str,
        dimension: int = 1024,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.dimension = dimension
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def availability(self):
        if not self.checkpoint_path.is_file():
            return False, f"Thiếu checkpoint Apple-CLIP: {self.checkpoint_path}"
        if importlib.util.find_spec("open_clip") is None:
            return False, "Thiếu package open_clip_torch; hãy cài requirements.txt."
        return True, "Model được tải lazy ở truy vấn Apple-CLIP đầu tiên."

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            return checkpoint
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint

    def _load(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            available, reason = self.availability()
            if not available:
                raise ModelUnavailableError(reason)
            try:
                import open_clip

                model, _, _ = open_clip.create_model_and_transforms(
                    self.MODEL_NAME,
                    pretrained=None,
                )
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
                state_dict = self._extract_state_dict(checkpoint)
                cleaned = {}
                for key, value in state_dict.items():
                    if key.startswith("module."):
                        key = key[len("module.") :]
                    if key.startswith("_orig_mod."):
                        key = key[len("_orig_mod.") :]
                    cleaned[key] = value
                model.load_state_dict(cleaned, strict=True)
                del checkpoint, state_dict, cleaned

                if self.device == "cuda":
                    model = model.to(device=self.device, dtype=torch.float16)
                else:
                    model = model.to(device=self.device)
                model.eval()
                tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
            except Exception as exc:
                raise ModelUnavailableError(
                    f"Không tải được Apple-CLIP finetune: {exc}"
                ) from exc
            self._model = model
            self._tokenizer = tokenizer

    def encode(self, query_text: str) -> np.ndarray:
        with self._inference_lock:
            self._load()
            tokens = self._tokenizer([query_text]).to(self.device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device,
                    dtype=torch.float16,
                    enabled=self.device == "cuda",
                ):
                    vector = self._model.encode_text(tokens)
                vector = torch.nn.functional.normalize(vector.float(), p=2, dim=-1)
        return normalize_query_vector(vector.cpu().numpy(), self.dimension)

    def unload(self):
        with self._inference_lock, self._load_lock:
            self._model = None
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class JinaEncoder:
    """Lazy multilingual text/image query encoder for the Jina retrieval space."""

    MODEL_ID = "jinaai/jina-embeddings-v5-omni-small"
    MODEL_REVISION = "05f4151c87083f204159bfa15e53fdb0320ffef1"

    def __init__(self, device: str, dimension: int = 1024):
        self.device = device
        self.dimension = dimension
        self._model = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def availability(self):
        if importlib.util.find_spec("sentence_transformers") is None:
            return False, "Thiếu package sentence-transformers; hãy cài requirements.txt."
        return True, "Model được tải lazy ở truy vấn Jina đầu tiên."

    def _load(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            available, reason = self.availability()
            if not available:
                raise ModelUnavailableError(reason)
            try:
                from sentence_transformers import SentenceTransformer

                loaded_model = SentenceTransformer(
                    self.MODEL_ID,
                    revision=self.MODEL_REVISION,
                    trust_remote_code=True,
                    model_kwargs={"default_task": "retrieval"},
                    device=self.device,
                )
                # transformers 4.57.x may fail AutoProcessor because the
                # standalone video config does not advertise its class. The
                # remote module silently falls back to processor=None, which
                # only breaks later when an image is encoded. Build the same
                # Qwen processor explicitly so text and image queries both work.
                transformer_module = loaded_model[0]
                if getattr(transformer_module, "processor", None) is None:
                    from transformers import (
                        Qwen2VLImageProcessor,
                        Qwen3VLProcessor,
                        Qwen3VLVideoProcessor,
                    )

                    common_kwargs = {
                        "revision": self.MODEL_REVISION,
                        "trust_remote_code": True,
                    }
                    image_processor = Qwen2VLImageProcessor.from_pretrained(
                        self.MODEL_ID,
                        min_pixels=262144,
                        max_pixels=1310720,
                        **common_kwargs,
                    )
                    video_processor = Qwen3VLVideoProcessor.from_pretrained(
                        self.MODEL_ID,
                        **common_kwargs,
                    )
                    transformer_module.processor = Qwen3VLProcessor(
                        image_processor=image_processor,
                        video_processor=video_processor,
                        tokenizer=transformer_module.tokenizer,
                    )
            except Exception as exc:
                raise ModelUnavailableError(f"Không tải được Jina encoder: {exc}") from exc
            self._model = loaded_model

    def _encode_queries(self, inputs) -> np.ndarray:
        with self._inference_lock:
            self._load()
            vectors = self._model.encode_query(
                inputs,
                normalize_embeddings=True,
                truncate_dim=self.dimension,
                convert_to_numpy=True,
            )
        return normalize_vectors(vectors, self.dimension)

    def encode(self, query_text: str) -> np.ndarray:
        return normalize_query_vector(self._encode_queries(query_text), self.dimension)

    def encode_texts(self, query_texts: list[str]) -> np.ndarray:
        return self._encode_queries(query_texts)

    def encode_image(self, image) -> np.ndarray:
        return normalize_query_vector(self._encode_queries(image), self.dimension)

    def unload(self):
        with self._inference_lock, self._load_lock:
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Kept so the offline caption encoder does not need a migration.
JinaTextEncoder = JinaEncoder
