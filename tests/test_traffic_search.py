import unittest

from traffic_search import expanded_query, normalize_text, parse_traffic_query


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


if __name__ == "__main__":
    unittest.main()
