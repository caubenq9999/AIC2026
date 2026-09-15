import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from semantic_search import AppleClipTextEncoder, JinaEncoder


class SemanticEncoderLifecycleTests(unittest.TestCase):
    def test_unload_is_noop_when_model_was_never_loaded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            apple = AppleClipTextEncoder(
                Path(temporary_directory) / "missing.pt", device="cpu"
            )
            jina = JinaEncoder(device="cpu")
            with patch("semantic_search.gc.collect") as collect:
                self.assertFalse(apple.unload())
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


if __name__ == "__main__":
    unittest.main()
