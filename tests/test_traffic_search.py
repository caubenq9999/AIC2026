import unittest
import csv
import gc
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from traffic_search import (
    TrafficSearchIndex,
    _artifact_key,
    expanded_query,
    merge_ranked_batches,
    normalize_text,
    parse_traffic_query,
)


class TrafficQueryTests(unittest.TestCase):
    def test_normalize_keeps_vietnamese_d_semantics(self):
        self.assertEqual(normalize_text("Đường đông, xe đỏ"), "duong dong xe do")

    def test_parse_vietnamese_vehicle_color_and_density(self):
        parsed = parse_traffic_query("Tìm xe máy màu đỏ trên đường đông xe")
        self.assertEqual(parsed["vehicles"], ["motorcycle"])
        self.assertEqual(parsed["colors"], ["red"])
        self.assertTrue(parsed["busy"])
        self.assertFalse(parsed["sparse"])

    def test_parse_multiple_vehicle_types(self):
        parsed = parse_traffic_query("ô tô và xe buýt màu trắng")
        self.assertEqual(parsed["vehicles"], ["car", "bus"])
        self.assertEqual(parsed["colors"], ["white"])

    def test_expansion_uses_readable_english_color(self):
        parsed = parse_traffic_query("xe tải màu bạc")
        expanded = expanded_query("xe tải màu bạc", parsed)
        self.assertIn("gray silver", expanded)
        self.assertIn("truck", expanded)

    def test_accent_collisions_do_not_create_false_colors(self):
        searching = parse_traffic_query("tìm xe máy")
        empty_road = parse_traffic_query("đường vắng xe")
        self.assertNotIn("purple", searching["colors"])
        self.assertNotIn("yellow_gold", empty_road["colors"])
        self.assertTrue(empty_road["sparse"])

    def test_explicit_yellow_and_purple_still_work(self):
        self.assertEqual(parse_traffic_query("xe màu vàng")["colors"], ["yellow_gold"])
        self.assertEqual(parse_traffic_query("ô tô màu tím")["colors"], ["purple"])

    def test_batch_rank_merge_balances_sources_without_raw_score_comparison(self):
        batch1 = [
            {"path": "L-first", "batch": "batch1", "score": 0.91},
            {"path": "L-second", "batch": "batch1", "score": 0.89},
        ]
        batch2 = [
            {"path": "N-first", "batch": "batch2", "score": 12.0},
            {"path": "N-second", "batch": "batch2", "score": -3.0},
        ]
        merged = merge_ranked_batches(batch1, batch2)
        self.assertEqual(
            [item["path"] for item in merged],
            ["L-first", "N-first", "L-second", "N-second"],
        )
        self.assertEqual(merged[0]["source_score"], 0.91)
        self.assertEqual(merged[1]["source_score"], 12.0)


class TrafficArtifactDiscoveryTests(unittest.TestCase):
    def test_standardized_shard_name_matches_legacy_detection_name(self):
        self.assertEqual(
            _artifact_key("N041-N050"),
            _artifact_key("Video_N041-N50_metadata.parquet"),
        )
        self.assertEqual(
            _artifact_key("N061-N080"),
            _artifact_key("Video_N061-N080_1fps_metadata.parquet"),
        )

    def test_embedding_only_shard_and_unextracted_keyframe_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "captions" / "S01"
            shard.mkdir(parents=True)
            vector = np.zeros((1, 1024), dtype=np.float32)
            vector[0, 0] = 1.0
            np.save(shard / "caption_embeddings.npy", vector)
            with (shard / "caption_mapping.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "row_index", "frame_id", "archive", "relative_path",
                    "parent_path", "frame_name", "caption", "hit_token_limit",
                ))
                writer.writeheader()
                writer.writerow({
                    "row_index": 0,
                    "relative_path": "odd/layout/S01_V001/frame_000025.webp",
                    "parent_path": "odd/layout/S01_V001",
                    "frame_name": "frame_000025.webp",
                    "caption": "A cyclist crosses the finish line.",
                })
            keyframe_root = root / "keyframes"
            keyframe_root.mkdir()
            with zipfile.ZipFile(keyframe_root / "Keyframes_S01.zip", "w") as archive:
                archive.writestr("another/layout/S01_V001/frame_000025.webp", b"fake-webp")
            media_root = root / "media-info"
            media_root.mkdir()
            (media_root / "S01_V001.json").write_text(
                json.dumps({"length": 10, "watch_url": "https://example.test/video"}),
                encoding="utf-8",
            )
            metadata_root = root / "metadata"
            metadata_root.mkdir()
            (metadata_root / "S01_V001.json").write_text(json.dumps([{
                "idx": 0,
                "video_id": "S01_V001",
                "frame_id": 25,
                "fps": 25.0,
                "frame_stamp": 1.0,
                "video_url": "https://example.test/metadata-video",
            }]), encoding="utf-8")

            index = TrafficSearchIndex(
                root / "captions", None, keyframe_root, None, media_root,
                metadata_dir=metadata_root,
            )
            self.assertEqual(index.size, 1)
            self.assertEqual(index.video_ids, ["S01_V001"])
            self.assertTrue(index.image_available[0])
            self.assertEqual(
                index.video_url_by_id["S01_V001"],
                "https://example.test/metadata-video",
            )
            self.assertEqual(index.timestamps[0], 1.0)
            self.assertEqual(index.fps_by_video["S01_V001"], 25.0)
            asset = index.image_asset_for_path(index.web_paths[0])
            self.assertEqual(asset["kind"], "bytes")
            self.assertEqual(asset["data"], b"fake-webp")
            del index
            gc.collect()


if __name__ == "__main__":
    unittest.main()
