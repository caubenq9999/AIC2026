"""Jina-only encoders and exact search over normalized NPY shards."""

from __future__ import annotations

import gc
import importlib.util
import threading
from pathlib import Path

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
