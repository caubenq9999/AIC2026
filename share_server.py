"""Lightweight LAN chat server for sharing AIC retrieval results."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(
    os.getenv("AIC_SHARE_DB", str(BASE_DIR / "share_messages.db"))
).expanduser().resolve()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/health": {"origins": "*"}})


def connect_database() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect_database() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                video_id TEXT,
                frame_n INTEGER,
                frame_idx INTEGER,
                pts_time REAL,
                image_path TEXT,
                query TEXT,
                search_mode TEXT,
                search_mode_label TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        # Giữ database từ bản chat cũ, chỉ bổ sung cột còn thiếu.
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(messages)")
        }
        if "search_mode" not in existing_columns:
            connection.execute("ALTER TABLE messages ADD COLUMN search_mode TEXT")
        if "search_mode_label" not in existing_columns:
            connection.execute("ALTER TABLE messages ADD COLUMN search_mode_label TEXT")


def clean_text(value, max_length: int) -> str:
    return str(value or "").strip()[:max_length]


def optional_int(value):
    if value in (None, "", "N/A"):
        return None
    return int(value)


def optional_float(value):
    if value in (None, "", "N/A"):
        return None
    return float(value)


def serialize_message(row: sqlite3.Row) -> dict:
    video = None
    if row["video_id"]:
        video = {
            "video_id": row["video_id"],
            "frame_n": row["frame_n"],
            "frame_idx": row["frame_idx"],
            "pts_time": row["pts_time"],
            "image_path": row["image_path"],
            "query": row["query"],
            "search_mode": row["search_mode"],
            "search_mode_label": row["search_mode_label"],
        }
    return {
        "id": row["id"],
        "sender": row["sender"],
        "text": row["text"],
        "video": video,
        "created_at": row["created_at"],
    }


@app.get("/health")
def health():
    with connect_database() as connection:
        message_count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    return jsonify({"status": "ok", "messages": message_count})


@app.get("/api/messages")
def get_messages():
    try:
        after = max(0, int(request.args.get("after", 0)))
        limit = min(200, max(1, int(request.args.get("limit", 100))))
    except ValueError:
        return jsonify({"error": "after và limit phải là số nguyên."}), 400

    with connect_database() as connection:
        if after:
            rows = connection.execute(
                "SELECT * FROM messages WHERE id > ? ORDER BY id ASC LIMIT ?",
                (after, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()[::-1]
    return jsonify({"messages": [serialize_message(row) for row in rows]})


@app.post("/api/messages")
def post_message():
    payload = request.get_json(silent=True) or {}
    sender = clean_text(payload.get("sender"), 40)
    text = clean_text(payload.get("text"), 2000)
    video = payload.get("video") if isinstance(payload.get("video"), dict) else {}
    video_id = clean_text(video.get("video_id"), 100)

    if not sender:
        return jsonify({"error": "Thiếu tên người gửi."}), 400
    if not text and not video_id:
        return jsonify({"error": "Tin nhắn hoặc video không được để trống."}), 400

    try:
        values = {
            "frame_n": optional_int(video.get("frame_n")),
            "frame_idx": optional_int(video.get("frame_idx")),
            "pts_time": optional_float(video.get("pts_time")),
        }
    except (TypeError, ValueError):
        return jsonify({"error": "Thông tin frame hoặc thời gian không hợp lệ."}), 400

    created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    with connect_database() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (
                sender, text, video_id, frame_n, frame_idx,
                pts_time, image_path, query, search_mode, search_mode_label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sender,
                text,
                video_id or None,
                values["frame_n"],
                values["frame_idx"],
                values["pts_time"],
                clean_text(video.get("image_path"), 500) or None,
                clean_text(video.get("query"), 1000) or None,
                clean_text(video.get("search_mode"), 80) or None,
                clean_text(video.get("search_mode_label"), 100) or None,
                created_at,
            ),
        )
        row = connection.execute(
            "SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return jsonify({"message": serialize_message(row)}), 201


@app.delete("/api/messages/<int:message_id>")
def delete_message(message_id: int):
    payload = request.get_json(silent=True) or {}
    sender = clean_text(payload.get("sender"), 40)
    if not sender:
        return jsonify({"error": "Thiếu tên người gửi."}), 400

    with connect_database() as connection:
        row = connection.execute(
            "SELECT sender FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            return jsonify({"error": "Tin nhắn không tồn tại."}), 404
        if row["sender"] != sender:
            return jsonify({"error": "Bạn chỉ có thể xóa tin nhắn của mình."}), 403
        connection.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    return jsonify({"status": "ok", "deleted_id": message_id})


@app.delete("/api/messages")
def delete_own_messages():
    payload = request.get_json(silent=True) or {}
    sender = clean_text(payload.get("sender"), 40)
    if not sender:
        return jsonify({"error": "Thiếu tên người gửi."}), 400

    with connect_database() as connection:
        cursor = connection.execute("DELETE FROM messages WHERE sender = ?", (sender,))
    return jsonify({"status": "ok", "deleted": cursor.rowcount})


initialize_database()


if __name__ == "__main__":
    port = int(os.getenv("AIC_SHARE_PORT", "5050"))
    print(f"Team chat: http://0.0.0.0:{port}")
    print(f"Database: {DATABASE_PATH}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
