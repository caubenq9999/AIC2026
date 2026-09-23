"""Offline contract checks: no request reaches a real DRES server."""

import unittest
from unittest.mock import Mock, patch

import requests

import dres_gateway


def response(status, body):
    item = Mock(status_code=status)
    item.json.return_value = body
    return item


class DresGatewayTests(unittest.TestCase):
    @patch("dres_gateway.requests.get")
    def test_check_active_evaluation(self, get):
        get.return_value = response(200, [
            {"id": "run-1", "name": "Final", "status": "ACTIVE"},
            {"id": "old", "name": "Old", "status": "TERMINATED"},
        ])
        result = dres_gateway.check_evaluations("secret-session", "run-1")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selected"]["id"], "run-1")
        self.assertEqual(len(result["active"]), 1)
        self.assertTrue(result["server"].startswith("https://"))
        self.assertNotIn("secret-session", str(result))
        self.assertEqual(get.call_args.kwargs["params"], {"session": "secret-session"})
        self.assertIs(get.call_args.kwargs["allow_redirects"], False)

    @patch("dres_gateway.requests.get")
    def test_check_rejects_inactive_and_wrong_session(self, get):
        get.return_value = response(200, [{"id": "old", "status": "TERMINATED"}])
        self.assertEqual(dres_gateway.check_evaluations("secret", "old")["status"], "error")
        get.return_value = response(401, {"status": False, "description": "Unauthorized"})
        self.assertEqual(dres_gateway.check_evaluations("secret", "old")["status"], "error")

    @patch("dres_gateway.requests.post")
    def test_200_with_wrong_verdict_is_accepted_not_correct(self, post):
        post.return_value = response(200, {"status": True, "submission": "WRONG", "description": "Wrong answer"})
        payload = {"answerSets": [{"answers": [{"mediaItemName": "L21_V001", "start": 1000, "end": 2000}]}]}
        result = dres_gateway.submit_answer("secret-session", "run-1", payload)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["verdict"], "WRONG")
        self.assertEqual(post.call_args.kwargs["json"], payload)
        self.assertEqual(post.call_args.kwargs["params"], {"session": "secret-session"})
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)
        self.assertNotIn("secret-session", str(result))

    @patch("dres_gateway.requests.post")
    def test_202_without_verdict_is_accepted(self, post):
        post.return_value = response(202, {"status": True, "submission": "INDETERMINATE", "description": "Queued"})
        result = dres_gateway.submit_answer("secret", "run", {"answerSets": [{}]})
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["remote_status"], 202)

    @patch("dres_gateway.requests.post")
    def test_412_is_rejected_and_timeout_is_unknown(self, post):
        post.return_value = response(412, {"status": False, "description": "Duplicate"})
        self.assertEqual(dres_gateway.submit_answer("secret", "run", {"answerSets": [{}]})["status"], "rejected")
        post.side_effect = requests.Timeout()
        self.assertEqual(dres_gateway.submit_answer("secret", "run", {"answerSets": [{}]})["status"], "unknown")

    @patch("dres_gateway.requests.post")
    def test_malformed_success_is_unknown(self, post):
        post.return_value = response(200, {"unexpected": True})
        self.assertEqual(dres_gateway.submit_answer("secret", "run", {"answerSets": [{}]})["status"], "unknown")

    @patch("dres_gateway.requests.post")
    def test_rejected_success_flag_and_error_description_redact_session(self, post):
        post.return_value = response(200, {"status": False, "description": "token secret-session rejected"})
        result = dres_gateway.submit_answer("secret-session", "run", {"answerSets": [{}]})
        self.assertEqual(result["status"], "rejected")
        self.assertNotIn("secret-session", str(result))

    @patch("dres_gateway.requests.post")
    def test_server_error_is_unknown_and_evaluation_is_path_encoded(self, post):
        post.return_value = response(500, {"status": False, "description": "Internal error"})
        result = dres_gateway.submit_answer("secret", "run /one", {"answerSets": [{}]})
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(post.call_args.args[0].endswith("/run%20%2Fone"))

    @patch.dict("dres_gateway.os.environ", {"BTC_API_BASE_URL": "http://example.com"})
    def test_remote_http_is_refused(self):
        with self.assertRaises(ValueError):
            dres_gateway.base_url()


if __name__ == "__main__":
    unittest.main()
