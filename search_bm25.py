"""Persistent inverted BM25 used by the OCR and ASR search branches.

The old implementation scanned every tokenized document once per query term.
This module stores document IDs and raw term frequencies in posting lists, so a
query only visits documents containing one of its terms.  Raw TF is retained so
changing k1/b does not require rebuilding the cache.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable, Iterable, Sequence

import numpy as np


CACHE_FORMAT_VERSION = 1
_REQUIRED_FILES = (
    "meta.json",
    "vocab.txt",
    "offsets.npy",
    "df.npy",
    "doclen.npy",
    "postings.npy",
    "tfs.npy",
)


def fingerprint_paths(paths: Iterable[str | Path]) -> dict:
    """Return a cheap deterministic signature for files/directories.

    Hashing multi-GB artifacts at every startup would defeat the cache.  Names,
    sizes and nanosecond mtimes are sufficient to invalidate locally managed
    OCR/ASR artifacts when they are replaced or edited.
    """

    entries = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.exists():
            entries.append([str(path), "missing"])
            continue
        if path.is_file():
            stat = path.stat()
            entries.append([str(path), stat.st_size, stat.st_mtime_ns])
            continue
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            stat = child.stat()
            entries.append(
                [
                    str(path),
                    child.relative_to(path).as_posix(),
                    stat.st_size,
                    stat.st_mtime_ns,
                ]
            )
    digest = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"sha256": digest, "entries": len(entries)}


class PersistentInvertedBM25:
    """BM25 over memory-mapped posting lists with a content-addressed cache."""

    def __init__(self, cache_dir: str | Path, k1: float = 1.5, b: float = 0.75):
        self.cache_dir = Path(cache_dir)
        with (self.cache_dir / "meta.json").open(encoding="utf-8") as stream:
            self.meta = json.load(stream)

        self.k1 = float(k1)
        self.b = float(b)
        self.doc_count = int(self.meta["doc_count"])
        self.avgdl = float(self.meta["avgdl"])
        self.offsets = np.load(self.cache_dir / "offsets.npy", mmap_mode="r")
        self.dfs = np.load(self.cache_dir / "df.npy", mmap_mode="r")
        self.doc_len = np.load(self.cache_dir / "doclen.npy", mmap_mode="r")
        self.postings = np.load(self.cache_dir / "postings.npy", mmap_mode="r")
        self.tfs = np.load(self.cache_dir / "tfs.npy", mmap_mode="r")
        with (self.cache_dir / "vocab.txt").open(encoding="utf-8") as stream:
            self.vocab = {term.rstrip("\n"): index for index, term in enumerate(stream)}

        self.idf_values = np.log(
            (self.doc_count - self.dfs.astype(np.float64) + 0.5)
            / (self.dfs.astype(np.float64) + 0.5)
            + 1.0
        )
        self.length_norm = self.k1 * (
            1.0
            - self.b
            + self.b * self.doc_len.astype(np.float64) / self.avgdl
        )

    @classmethod
    def load_or_build(
        cls,
        cache_root: str | Path,
        name: str,
        doc_count: int,
        tokens_for_document: Callable[[int], Sequence[str]],
        source_signature: dict,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "PersistentInvertedBM25":
        signature = {
            "format": CACHE_FORMAT_VERSION,
            "name": str(name),
            "doc_count": int(doc_count),
            "source": source_signature,
        }
        signature_json = json.dumps(
            signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        cache_key = hashlib.sha256(signature_json.encode("utf-8")).hexdigest()[:20]
        cache_dir = Path(cache_root) / str(name) / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not cls._is_valid_cache(cache_dir, signature):
            cls._build_with_lock(
                cache_dir,
                signature,
                doc_count=int(doc_count),
                tokens_for_document=tokens_for_document,
            )
        return cls(cache_dir, k1=k1, b=b)

    @staticmethod
    def _is_valid_cache(cache_dir: Path, signature: dict) -> bool:
        if not all((cache_dir / filename).is_file() for filename in _REQUIRED_FILES):
            return False
        try:
            with (cache_dir / "meta.json").open(encoding="utf-8") as stream:
                meta = json.load(stream)
            if meta.get("signature") != signature:
                return False
            vocab_size = int(meta["vocab_size"])
            nnz = int(meta["nnz"])
            doc_count = int(meta["doc_count"])
            return (
                np.load(cache_dir / "offsets.npy", mmap_mode="r").shape
                == (vocab_size + 1,)
                and np.load(cache_dir / "df.npy", mmap_mode="r").shape
                == (vocab_size,)
                and np.load(cache_dir / "doclen.npy", mmap_mode="r").shape
                == (doc_count,)
                and np.load(cache_dir / "postings.npy", mmap_mode="r").shape
                == (nnz,)
                and np.load(cache_dir / "tfs.npy", mmap_mode="r").shape
                == (nnz,)
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    @classmethod
    def _build_with_lock(
        cls,
        cache_dir: Path,
        signature: dict,
        doc_count: int,
        tokens_for_document: Callable[[int], Sequence[str]],
    ) -> None:
        lock_path = cache_dir / ".build.lock"
        deadline = time.monotonic() + 1800.0
        while True:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
                owns_lock = True
                break
            except FileExistsError:
                if cls._is_valid_cache(cache_dir, signature):
                    return
                try:
                    if time.time() - lock_path.stat().st_mtime > 3600:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out while building BM25 cache: {lock_path}"
                    )
                time.sleep(0.5)

        try:
            if cls._is_valid_cache(cache_dir, signature):
                return
            cls._build(cache_dir, signature, doc_count, tokens_for_document)
        finally:
            if owns_lock:
                lock_path.unlink(missing_ok=True)

    @staticmethod
    def _build(
        cache_dir: Path,
        signature: dict,
        doc_count: int,
        tokens_for_document: Callable[[int], Sequence[str]],
    ) -> None:
        if doc_count <= 0:
            raise ValueError("BM25 requires at least one document.")

        print(f"[BM25:{signature['name']}] Building cache pass 1/2...", flush=True)
        document_frequency: dict[str, int] = {}
        doclen = np.empty(doc_count, dtype=np.int32)
        for document_index in range(doc_count):
            tokens = list(tokens_for_document(document_index))
            doclen[document_index] = len(tokens)
            for term in set(tokens):
                document_frequency[term] = document_frequency.get(term, 0) + 1
            if (document_index + 1) % 100_000 == 0:
                print(f"  {document_index + 1:,}/{doc_count:,} documents", flush=True)

        terms = sorted(document_frequency)
        vocab = {term: index for index, term in enumerate(terms)}
        dfs = np.fromiter(
            (document_frequency[term] for term in terms),
            dtype=np.int32,
            count=len(terms),
        )
        del document_frequency

        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        np.cumsum(dfs, out=offsets[1:])
        nnz = int(offsets[-1])
        cursor = offsets[:-1].copy()

        temp_postings = cache_dir / "postings.tmp.npy"
        temp_tfs = cache_dir / "tfs.tmp.npy"
        postings = np.lib.format.open_memmap(
            temp_postings, mode="w+", dtype=np.int32, shape=(nnz,)
        )
        # Keep raw TF lossless so the persistent implementation is exactly
        # equivalent to the previous in-memory scorer, even for unusual long
        # OCR/ASR documents.
        tfs = np.lib.format.open_memmap(
            temp_tfs, mode="w+", dtype=np.uint32, shape=(nnz,)
        )

        print(f"[BM25:{signature['name']}] Building cache pass 2/2...", flush=True)
        for document_index in range(doc_count):
            counts = collections.Counter(tokens_for_document(document_index))
            for term, frequency in counts.items():
                term_index = vocab[term]
                position = int(cursor[term_index])
                postings[position] = document_index
                tfs[position] = int(frequency)
                cursor[term_index] = position + 1
            if (document_index + 1) % 100_000 == 0:
                print(f"  {document_index + 1:,}/{doc_count:,} documents", flush=True)

        postings.flush()
        tfs.flush()
        del postings, tfs, cursor, vocab

        temp_files = {
            "postings.npy": temp_postings,
            "tfs.npy": temp_tfs,
            "offsets.npy": cache_dir / "offsets.tmp.npy",
            "df.npy": cache_dir / "df.tmp.npy",
            "doclen.npy": cache_dir / "doclen.tmp.npy",
            "vocab.txt": cache_dir / "vocab.tmp.txt",
        }
        np.save(temp_files["offsets.npy"], offsets)
        np.save(temp_files["df.npy"], dfs)
        np.save(temp_files["doclen.npy"], doclen)
        with temp_files["vocab.txt"].open("w", encoding="utf-8", newline="\n") as stream:
            for term in terms:
                stream.write(term + "\n")

        for final_name, temp_path in temp_files.items():
            os.replace(temp_path, cache_dir / final_name)

        meta = {
            "signature": signature,
            "doc_count": doc_count,
            "vocab_size": len(terms),
            "nnz": nnz,
            "avgdl": float(doclen.mean()),
        }
        temp_meta = cache_dir / "meta.tmp.json"
        temp_meta.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temp_meta, cache_dir / "meta.json")
        print(
            f"[BM25:{signature['name']}] Cache ready: "
            f"{doc_count:,} documents, {len(terms):,} terms, {nnz:,} postings.",
            flush=True,
        )

    def has_term(self, term: str) -> bool:
        return term in self.vocab

    def idf_for(self, term: str) -> float:
        return float(self.idf_values[self.vocab[term]])

    def _score_terms(self, query: Sequence[str], distinct: bool):
        terms = list(dict.fromkeys(query)) if distinct else list(query)
        scores = np.zeros(self.doc_count, dtype=np.float64)
        match_counts = (
            np.zeros(self.doc_count, dtype=np.uint16) if distinct else None
        )
        for term in terms:
            term_index = self.vocab.get(term)
            if term_index is None:
                continue
            start = int(self.offsets[term_index])
            end = int(self.offsets[term_index + 1])
            document_ids = np.asarray(self.postings[start:end], dtype=np.int32)
            term_frequencies = np.asarray(self.tfs[start:end], dtype=np.float64)
            numerator = term_frequencies * (self.k1 + 1.0)
            denominator = term_frequencies + self.length_norm[document_ids]
            scores[document_ids] += (
                self.idf_values[term_index] * numerator / denominator
            )
            if match_counts is not None:
                match_counts[document_ids] += 1
        return scores, match_counts

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        scores, _ = self._score_terms(query, distinct=False)
        return scores

    def get_scores_with_match_counts(
        self, query: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        scores, match_counts = self._score_terms(query, distinct=True)
        return scores, match_counts

    def close(self) -> None:
        """Release mmap handles explicitly (important for cache cleanup on Windows)."""
        for name in ("offsets", "dfs", "doc_len", "postings", "tfs"):
            array = getattr(self, name, None)
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
