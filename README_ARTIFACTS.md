# Chuẩn tổ chức artifacts thống nhất

MFusion-VR dùng **một chuẩn artifact chung** cho toàn bộ collection `L`, `M`, `N`
và `S`. Batch 1/Batch 2 chỉ là tên các đợt phát hành dữ liệu; chúng không tạo ra
hai layout khác nhau trong hệ thống.

Apple-CLIP fine-tuned đã được loại khỏi luồng chạy chính. Chuẩn mới chỉ dùng Jina
cho image/caption retrieval. Folder Apple cũ không còn cần thiết để khởi động app.

## 1. Cấu trúc chuẩn

Toàn bộ dữ liệu lớn nằm trong một root `artifacts/`:

```text
AIC2025/
├── app.py
├── artifacts-manifest.json
├── README_ARTIFACTS.md
│
└── artifacts/
    ├── embeddings/
    │   └── jina/
    │       ├── image/
    │       │   ├── L21.npy
    │       │   ├── L22.npy
    │       │   ├── ...
    │       │   └── L30.npy
    │       └── caption/
    │           ├── L21.npy
    │           ├── L21.json
    │           ├── ...
    │           ├── L30.npy
    │           ├── L30.json
    │           ├── M01/
    │           │   ├── caption_embeddings.npy
    │           │   └── caption_mapping.csv
    │           ├── ...
    │           ├── N001-N010/
    │           │   ├── caption_embeddings.npy
    │           │   └── caption_mapping.csv
    │           └── S01/
    │               ├── caption_embeddings.npy
    │               └── caption_mapping.csv
    │
    ├── metadata/
    │   ├── frames/
    │   │   ├── L21_V001.json
    │   │   ├── ...
    │   │   ├── M01_V001.json
    │   │   ├── N001-V001.json
    │   │   └── S01-V001.json
    │   └── asr/
    │       ├── L21_V001.json
    │       └── ...
    │
    ├── keyframes/
    │   ├── L21/L21_V001/000000.webp
    │   ├── ...
    │   ├── Keyframes_M01.zip
    │   ├── Keyframes_N001-N010.zip
    │   └── Keyframes_S01.zip
    │
    ├── detections/
    │   ├── N001-N010.parquet
    │   └── ...
    │
    └── videos/
        ├── M01_V001.mp4
        ├── N001-V001.mp4
        └── ...
```

App tự ưu tiên cấu trúc này khi `artifacts/` tồn tại. Trong giai đoạn chuyển đổi,
layout cũ ở root repo vẫn được hỗ trợ để không buộc phải di chuyển tất cả dữ liệu
trong một lần.

## 2. Quy ước collection

`image/` chứa các shard ảnh phẳng `L21.npy`–`L30.npy`. `caption/` chứa các shard
caption phẳng `L21.npy`–`L30.npy` cùng JSON sidecar; các đợt M/N/S giữ dạng folder
đã chuẩn hóa như `M01/` hoặc `N001-N010/` vì mỗi shard cần cả vector và mapping.

Collection logic của nhóm M là đủ `M01`–`M10`. Tên folder như `M08-M09`
chỉ là **shard vật lý** gộp embedding của hai collection `M08` và `M09`;
nó không phải tên một collection mới. Tương tự, `N001-N010` là shard gộp
nhiều collection N.

Các folder M/N/S dùng cùng tên file:

| File | Vai trò | Bắt buộc |
|---|---|---:|
| `caption_embeddings.npy` | Vector caption Jina | Có |
| `caption_mapping.csv` | Ánh xạ từng dòng vector về video/frame/caption | Có |
| `embedding_manifest.json` | Model, dimension, count, checksum | Khuyến nghị |

Hiện `L21–L30` có cả image và caption embedding. Các collection `M/N/S` mới chỉ
có caption embedding vẫn hợp lệ. Không cần đổi `L21.npy` thành folder `L21/`.

## 3. Hợp đồng embedding và mapping

### Embedding

- Shape: `(N, 1024)`.
- Các dòng phải được L2-normalize.
- Số dòng phải bằng số record tương ứng trong mapping/metadata.
- Không sort riêng NPY hoặc CSV sau khi encode.

### `caption_mapping.csv`

CSV cần đủ thông tin để xác định từng dòng:

```csv
row_index,relative_path,parent_path,frame_name,caption
0,Keyframes/M01_V001/000000.webp,Keyframes/M01_V001,000000.webp,"..."
```

Tên cột phụ như `archive`, `frame_id`, `hit_token_limit` được phép giữ lại.
`video_id` và frame number phải suy được từ `relative_path`, hoặc từ
`parent_path + frame_name`.

Với `L21–L30`, loader ánh xạ theo thứ tự metadata toàn cục và kiểm tra số dòng,
thứ tự collection. JSON sidecar `L21.json`–`L30.json` mô tả rows, dimension,
model và checksum nguồn; không cần tạo `caption_mapping.csv` cho các shard này.

## 4. Metadata dùng chung

Tất cả metadata keyframe đặt chung trong:

```text
artifacts/metadata/frames/
```

Không còn khái niệm “metadata Batch 1” và “metadata Batch 2”. Một file đại diện
cho một video, ví dụ `L21_V001.json`, `M01_V001.json`, `N001-V001.json`.

Các trường chuẩn:

```json
{
  "video_id": "M01_V001",
  "frame_id": 11,
  "frame_idx": 11,
  "fps": 25.0,
  "frame_stamp": 0.44,
  "path": "Keyframes/M01_V001/000011.webp",
  "video_url": "https://youtube.com/watch?v=...",
  "ocr_text": ""
}
```

Quy tắc:

- `video_id + frame_id` phải duy nhất.
- `frame_stamp` hoặc `pts_time` là timestamp tính bằng giây.
- `ocr_text` được phép rỗng nếu collection chưa chạy OCR.
- `video_url` được phép rỗng nếu chưa có video local/YouTube.
- `idx` cũ có thể giữ lại nhưng không dùng làm khóa chung giữa các collection.
- Với vector L21–L30 legacy, `idx` vẫn phải đúng thứ tự cũ cho đến khi hoàn tất
  migration sang mapping CSV.

App tách record theo `video_id`, vì vậy `idx=0` xuất hiện lại ở collection mới
không còn gây lỗi trùng global.

## 5. Keyframe

Tất cả keyframe đặt dưới một root:

```text
artifacts/keyframes/
```

Chấp nhận folder đã giải nén, ZIP gốc của BTC và nhiều lớp trung gian như
`output/keyframes/...`.

Loader dò theo `video_id + frame number`, nên cả hai path sau đều hợp lệ:

```text
L21/L21_V001/000020.webp
output/keyframes/N001-V001/000025.webp
```

Không cần tải đủ keyframe để test retrieval. Nếu một shard mới chỉ có embedding,
UI dùng placeholder nhưng vẫn mở detail và nộp được.

## 6. OCR, ASR và detection

OCR đã nằm ngay trong trường `ocr_text` của metadata frame. ASR cũng
thuộc artifact metadata chung, nhưng được tách thành các segment có timestamp:

```text
artifacts/metadata/frames/  # metadata keyframe, bao gồm ocr_text
artifacts/metadata/asr/     # transcript ASR theo video và khoảng thời gian
artifacts/detections/
```

Vì OCR và ASR đã nằm trong gói metadata nên không cần tạo artifact
`ocr/` hay `asr/` độc lập cho layout mới. Detection vẫn là enrichment riêng.

## 7. Video local và YouTube

Video local đặt tại `artifacts/videos/`. Backend quét đệ quy các định dạng
`.mp4`, `.webm`, `.mov`, `.m4v` và các file ZIP chứa chúng; tên file video bên
trong phải trùng `video_id`.

```text
artifacts/videos/M01_V001.mp4
artifacts/videos/N001-V001.mp4
artifacts/videos/Videos_M01.zip  # có thể chứa videos/M01_V001.mp4, ...
```

Không bắt buộc giải nén ZIP video. Backend index member và stream bằng HTTP Range;
nếu cùng một video tồn tại cả dạng giải nén và trong ZIP thì file giải nén được
ưu tiên.

Thứ tự chọn nguồn phát:

1. video local đã giải nén;
2. `video_url` trong metadata;
3. không mở player, nhưng detail và submission vẫn hoạt động.

FPS, timestamp, URL YouTube và thông tin video đều thuộc metadata chung. Không
tạo thêm `media-info/` trong layout chuẩn. Folder `aic26-b2-media-info/` cũ chỉ
được backend đọc tạm để bù URL cho metadata legacy chưa hoàn chỉnh.

## 8. Manifest

`artifacts-manifest.json` ở repo chỉ chứa metadata nhỏ: model/revision Jina,
dimension, collection logic, shard vật lý, số dòng và schema version. Manifest
schema 2 đã dùng chung cho L/M/N và không còn đường dẫn Apple-CLIP.

## 9. Chuyển layout hiện tại sang chuẩn mới

| Hiện tại | Chuẩn mới |
|---|---|
| `embedding/jina/jina_embeddings_npy/L21.npy` | `artifacts/embeddings/jina/image/L21.npy` |
| `embedding/jina/caption_embeddings_npy/L21.npy` | `artifacts/embeddings/jina/caption/L21.npy` |
| `embedding/jina/caption_embeddings_npy/L21.json` | `artifacts/embeddings/jina/caption/L21.json` |
| `captionbatch2_emb/M01/*` | `artifacts/embeddings/jina/caption/M01/*` |
| `captionbatch2_emb/N001-N010/*` | `artifacts/embeddings/jina/caption/N001-N010/*` |
| `ocr/metadata_ocr_filtered/metadata/*.json` | `artifacts/metadata/frames/*.json` |
| `keyframes/*` | `artifacts/keyframes/*` |
| `asr/metadata_asr_clean/*` | `artifacts/metadata/asr/*` |
| `detection segmentation/detection segmentation/*` | `artifacts/detections/*` |
| `videos/*` | `artifacts/videos/*` |

Không cần copy cả hai nơi. Sau khi xác nhận app đọc chuẩn mới, có thể di chuyển
artifact cũ hoặc dùng junction/symlink để tránh nhân đôi hàng trăm GB.

Folder `embedding/apple_finetuned/` không còn được app sử dụng. Tài liệu này không
tự động yêu cầu xóa nó; chỉ xóa sau khi đội xác nhận không cần giữ checkpoint cho
thử nghiệm khác.

## 10. Cấu hình

Root mặc định là `artifacts/`. Có thể đổi toàn bộ root bằng một biến:

```powershell
$env:AIC_ARTIFACTS_DIR = "D:\AIC_DATA\artifacts"
.\.venv\Scripts\python.exe app.py
```

Các biến override chi tiết vẫn được hỗ trợ khi cần:

```powershell
$env:AIC_JINA_EMBEDDINGS_DIR = "D:\AIC_DATA\artifacts\embeddings\jina"
$env:AIC_KEYFRAMES_DIR = "D:\AIC_DATA\artifacts\keyframes"
$env:AIC_VIDEOS_DIR = "D:\AIC_DATA\artifacts\videos"
$env:AIC_OCR_METADATA_PATH = "D:\AIC_DATA\artifacts\metadata\frames"
$env:AIC_ASR_METADATA_DIR = "D:\AIC_DATA\artifacts\metadata\asr"
$env:AIC_TRAFFIC_DETECTION_PATH = "D:\AIC_DATA\artifacts\detections"
```

Đường dẫn legacy vẫn được tự dò nếu `artifacts/` chưa tồn tại.

## 11. File nào được push

Nên push source code, README, `requirements.txt`, manifest nhỏ và script kiểm
tra/migration.

Không push `artifacts/`, `.cache/`, NPY/NPZ, checkpoint, video, keyframe, ZIP,
Parquet hoặc index/cache sinh tự động.

## 12. Checklist thêm collection

1. Tạo một folder M/N/S trong `artifacts/embeddings/jina/caption/`.
2. Đặt `caption_embeddings.npy` và `caption_mapping.csv` cùng folder.
3. Kiểm tra số dòng khớp và dimension bằng 1,024.
4. Chép metadata frame (có `ocr_text`) vào `artifacts/metadata/frames/`
   và ASR vào `artifacts/metadata/asr/`.
5. Thêm keyframe ZIP/folder nếu cần thumbnail thật.
6. Thêm MP4/ZIP video hoặc điền `video_url` vào metadata nếu cần phát video.
7. Detection có thể bổ sung sau.
8. Restart backend và kiểm tra:

```powershell
Invoke-RestMethod http://localhost:5000/health | ConvertTo-Json -Depth 6
```

9. Hard refresh trình duyệt sau khi frontend thay đổi.
