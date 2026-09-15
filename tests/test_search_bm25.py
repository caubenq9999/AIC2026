import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from search_bm25 import PersistentInvertedBM25


class ReferenceBM25:
    """Previous full-corpus implementation, kept only for equivalence tests."""

    def __init__(self, corpus, k1=1.5, b=0.20):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(document) for document in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.doc_count = len(corpus)
        document_frequencies = {}
        for document in corpus:
            for term in set(document):
                document_frequencies[term] = document_frequencies.get(term, 0) + 1
        self.idf = {
            term: math.log(
                (self.doc_count - frequency + 0.5) / (frequency + 0.5) + 1.0
            )
            for term, frequency in document_frequencies.items()
        }

    def scores_with_counts(self, query):
        scores = np.zeros(self.doc_count)
        counts = np.zeros(self.doc_count, dtype=np.uint16)
        for term in dict.fromkeys(query):
            if term not in self.idf:
                continue
            frequencies = np.asarray(
                [document.count(term) for document in self.corpus], dtype=np.float64
            )
            counts += frequencies > 0
            numerator = frequencies * (self.k1 + 1)
            denominator = frequencies + self.k1 * (
                1 - self.b + self.b * np.asarray(self.doc_len) / self.avgdl
            )
            scores += self.idf[term] * numerator / denominator
        return scores, counts


class PersistentInvertedBM25Tests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "bắc bộ địa lý bắc".split(),
            "bắc".split(),
            "bài giảng địa lý vùng bắc bộ nhiều chữ".split(),
            "hồ tùng mậu".split(),
            "hoàn toàn không liên quan".split(),
        ]

    def test_scores_and_match_counts_equal_previous_implementation(self):
        reference = ReferenceBM25(self.corpus)
        with tempfile.TemporaryDirectory() as temporary_directory:
            index = PersistentInvertedBM25.load_or_build(
                temporary_directory,
                "test",
                len(self.corpus),
                lambda document_index: self.corpus[document_index],
                {"fixture": "v1"},
                k1=1.5,
                b=0.20,
            )
            for query in (
                ["bắc", "bộ"],
                ["bắc", "bắc", "bộ"],
                ["hồ", "tùng", "mậu"],
                ["không-có-trong-index"],
            ):
                expected_scores, expected_counts = reference.scores_with_counts(query)
                scores, counts = index.get_scores_with_match_counts(query)
                np.testing.assert_allclose(scores, expected_scores, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(counts, expected_counts)
            index.close()

    def test_valid_cache_is_reused_without_reading_corpus(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            signature = {"fixture": "reuse-v1"}
            initial = PersistentInvertedBM25.load_or_build(
                temporary_directory,
                "test",
                len(self.corpus),
                lambda document_index: self.corpus[document_index],
                signature,
            )
            initial.close()

            def fail_if_called(_document_index):
                raise AssertionError("A valid persistent index must not rebuild")

            cached = PersistentInvertedBM25.load_or_build(
                temporary_directory,
                "test",
                len(self.corpus),
                fail_if_called,
                signature,
            )
            self.assertEqual(cached.doc_count, len(self.corpus))
            self.assertTrue(cached.has_term("bắc"))
            cached.close()


if __name__ == "__main__":
    unittest.main()
