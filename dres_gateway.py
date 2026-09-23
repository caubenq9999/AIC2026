"""Small, testable DRES v2 client for final-round submissions.

Never log session tokens, request URLs (which contain a session query parameter),
or complete payloads here. A transport failure after POST is *uncertain*, not a
safe invitation to retry the same submission.
"""

import os
from urllib.parse import quote, urlsplit

import requests


DEFAULT_DRES_BASE_URL = "https://eventretrieval.oj.io.vn"
CONNECT_TIMEOUT = 3
READ_TIMEOUT = 12


def base_url():
    value = os.getenv("BTC_API_BASE_URL", DEFAULT_DRES_BASE_URL).strip().rstrip("/")
    parsed = urlsplit(value)
    local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
    if (parsed.scheme != "https" and not local_http) or not parsed.hostname or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("BTC_API_BASE_URL phải là HTTPS origin (HTTP chỉ cho localhost).")
    return value


def _description(body, fallback, secret=""):
    if isinstance(body, dict):
        value = body.get("description") or body.get("message")
        if isinstance(value, str) and value.strip():
            message = value.strip()[:300]
            return message.replace(secret, "[redacted]") if secret else message
    return fallback


def _json_or_none(response):
    try:
        return response.json()
    except ValueError:
        return None


def check_evaluations(session_id, evaluation_id=""):
    """Read only: GET the evaluations visible to this DRES session."""
    if not session_id:
        return {"status": "error", "message": "Thiếu sessionID."}
    try:
        server = base_url()
        response = requests.get(
            f"{server}/api/v2/client/evaluation/list",
            params={"session": session_id},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=False,
        )
    except (requests.RequestException, ValueError):
        return {"status": "error", "message": "Không kết nối được DRES; kiểm tra địa chỉ hoặc mạng."}
    body = _json_or_none(response)
    if response.status_code != 200:
        message = "Session không hợp lệ hoặc hết hạn." if response.status_code == 401 else _description(body, f"DRES trả HTTP {response.status_code}.", session_id)
        return {"status": "error", "message": message}
    if not isinstance(body, list):
        return {"status": "error", "message": "DRES trả danh sách evaluation không hợp lệ."}
    evaluations = [
        {"id": item["id"], "name": str(item.get("name") or ""), "status": str(item.get("status") or "")}
        for item in body if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    active = [item for item in evaluations if item["status"] == "ACTIVE"]
    selected = next((item for item in evaluations if item["id"] == evaluation_id), None)
    if evaluation_id and not selected:
        return {"status": "error", "message": "Evaluation ID không có trong danh sách của session này.", "active": active}
    if selected and selected["status"] != "ACTIVE":
        return {"status": "error", "message": "Evaluation đã chọn không ở trạng thái ACTIVE.", "active": active}
    return {"status": "ok", "server": server, "active": active, "selected": selected}


def submit_answer(session_id, evaluation_id, answer_payload):
    """POST once. The caller must never automatically retry an unknown result."""
    if not session_id or not evaluation_id:
        return {"status": "rejected", "message": "Thiếu sessionID hoặc evaluationID."}
    if not isinstance(answer_payload, dict) or not answer_payload.get("answerSets"):
        return {"status": "rejected", "message": "Gói đáp án không hợp lệ."}
    try:
        response = requests.post(
            f"{base_url()}/api/v2/submit/{quote(evaluation_id, safe='')}",
            params={"session": session_id},
            json=answer_payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=False,
        )
    except ValueError:
        return {"status": "rejected", "message": "BTC_API_BASE_URL chưa hợp lệ."}
    except requests.RequestException:
        return {"status": "unknown", "message": "Mất kết nối hoặc quá thời gian chờ; DRES có thể đã nhận bài. Kiểm tra trên DRES trước khi gửi lại."}

    body = _json_or_none(response)
    if response.status_code in (200, 202):
        if isinstance(body, dict) and body.get("status") is False:
            return {"status": "rejected", "message": _description(body, "DRES từ chối bài nộp.", session_id)}
        if not isinstance(body, dict) or body.get("status") is not True:
            return {"status": "unknown", "message": "DRES trả phản hồi chưa thể xác nhận đã nhận bài. Kiểm tra trên DRES trước khi gửi lại."}
        verdict = body.get("submission")
        verdict = verdict if verdict in {"CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"} else None
        return {
            "status": "accepted",
            "verdict": verdict,
            "remote_status": response.status_code,
            "message": _description(body, "DRES đã nhận bài.", session_id),
        }
    if response.status_code in (400, 401, 404, 412):
        return {
            "status": "rejected",
            "remote_status": response.status_code,
            "message": _description(body, f"DRES từ chối bài nộp (HTTP {response.status_code}).", session_id),
        }
    return {
        "status": "unknown",
        "remote_status": response.status_code,
        "message": "DRES trả trạng thái không rõ; kiểm tra trên DRES trước khi gửi lại.",
    }
