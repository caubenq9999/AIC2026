import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from semantic_search import JinaEncoder, ShardedNpyIndex


class SemanticEncoderLifecycleTests(unittest.TestCase):
    def test_unload_is_noop_when_model_was_never_loaded(self):
        jina = JinaEncoder(device="cpu")
        with patch("semantic_search.gc.collect") as collect:
            self.assertFalse(jina.unload())
            collect.assert_not_called()

    def test_loaded_model_is_released_once(self):
        jina = JinaEncoder(device="cpu")
        jina._model = object()
        with patch("semantic_search.gc.collect") as collect:
            self.assertTrue(jina.unload())
            self.assertFalse(jina.is_loaded)
            self.assertFalse(jina.unload())
            collect.assert_called_once_with()


class UnifiedArtifactLayoutTests(unittest.TestCase):
    def test_collection_folder_uses_standard_embedding_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            collection = root / "L21"
            collection.mkdir()
            np.save(
                collection / "caption_embeddings.npy",
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
            records = [
                {"video_id": "L21_V001", "path": "L21/L21_V001/000001.webp"},
                {"video_id": "L21_V001", "path": "L21/L21_V001/000002.webp"},
            ]

            index = ShardedNpyIndex(
                "caption",
                root,
                records,
                expected_dimension=2,
                shard_filename="caption_embeddings.npy",
            )

            scores, indices = index.search(
                np.asarray([[1.0, 0.0]], dtype=np.float32), top_k=1
            )
            self.assertEqual(index.ntotal, 2)
            self.assertEqual(int(indices[0, 0]), 0)
            self.assertAlmostEqual(float(scores[0, 0]), 1.0)
            index.shards.clear()
            del index


if __name__ == "__main__":
    unittest.main()
