# MFusion Team Chat

Chat nội bộ để thành viên gửi tin nhắn và chia sẻ keyframe, `video_id`, frame, timestamp, query và chế độ search đã sinh ra frame trong lúc thi.

## Thay đổi so với code captain

### 5 file mới

- `share_server.py`: chat server Flask + SQLite, chạy trên máy host.
- `share_client.js`: gửi/nhận/xóa tin và chia sẻ keyframe.
- `share_client.css`: giao diện popup kiểu Messenger.
- `run_share_server.bat`: khởi động chat server bằng một lệnh.
- `READMEteamchat.md`: tài liệu này.

`share_messages.db` được tạo tự động khi chạy và **không push lên Git**.

### Các file có sẵn được thay đổi

- `app.py`: cho phép serve `share_client.js` và `share_client.css`.
- `index.html`: thêm popup chat và import hai asset chat.
- `script.js`: lưu query + chế độ search theo từng kết quả để gửi đúng nguồn gốc của frame.
- `.gitignore`: bỏ qua database chat và các file SQLite tạm.

Database chat cũ được tự động thêm hai cột search mode khi khởi động lại `share_server.py`; tin nhắn cũ không bị xóa.

## Cách chạy và join

### Máy host

```powershell
# Terminal 1: Video Retrieval
python app.py

# Terminal 2: Team Chat
.\run_share_server.bat
```

### Máy thành viên

Chỉ chạy Video Retrieval local:

```powershell
python app.py
```

Mở `http://127.0.0.1:5000`, bấm popup chat → ⚙, nhập tên và địa chỉ:

```text
http://IP-MAY-HOST:5050
```

Ví dụ cùng LAN: `http://192.168.1.18:5050`. Nếu dùng Radmin VPN, nhập IP Radmin của host, thường có dạng `26.x.x.x:5050`.

Kiểm tra kết nối từ máy thành viên:

```text
http://IP-MAY-HOST:5050/health
```

Thấy `{"status":"ok", ...}` là join thành công. Host phải giữ terminal chat mở và Windows Firewall phải cho phép TCP port `5050`.
