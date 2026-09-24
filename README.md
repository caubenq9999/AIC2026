# AIC Video Retrieval — Jina + Apple-CLIP Finetune

Ứng dụng Flask tìm keyframe cho AIC/VBS. Semantic retrieval hỗ trợ Jina v5 và
Apple-CLIP `ViT-H-14-378-quickgelu` đã finetune. Jina được pin tại revision
`05f4151c87083f204159bfa15e53fdb0320ffef1`.

Các chế độ trên giao diện:

- **Semantic · Apple-CLIP**: text → Apple-CLIP image vectors đã finetune.
- **Semantic · Jina**: text → Jina image vectors.
- **Jina · Hybrid**: gộp rank từ Jina image vectors và Jina caption vectors bằng RRF.
- **OCR** và **ASR**: BM25 giảm length penalty, sau đó rerank theo độ phủ từ
  khóa/cụm từ để văn bản hoặc transcript dài không bị lép vế vô lý.
- **Fusion**: Jina Hybrid + OCR + ASR, có trọng số riêng cho từng nhánh.
- **Tìm ảnh tương tự**: ảnh → Jina image vectors, tùy chọn YOLO Auto-Crop.
- **TRAKE text**: Jina Hybrid retrieval rồi temporal alignment theo thứ tự sự kiện.
- **Tìm giao ảnh**: nhiều ảnh → Jina image retrieval rồi giao trong cửa sổ frame.

Thanh công cụ ở đầu trang giữ cố định khi cuộn: chọn loại câu hỏi **KIS / QA /
TRAKE** (mặc định KIS), phương thức tìm kiếm, số kết quả và tùy chọn nhóm theo
video. Lựa chọn loại câu hỏi được giữ khi chuyển giữa các frame; sau khi gửi,
dữ liệu đang nhập được giữ để đối chiếu phản hồi. Nút **Sáng/Tối** trong header
lưu lựa chọn giao diện trên trình duyệt. Trong thẻ chi tiết ảnh, Video, Giây và
`frame_idx` nằm cùng một hàng. Dải frame lân cận
tải theo cửa sổ ±15 frame, dùng thumbnail lớn hơn; có thể lăn chuột, kéo ngang,
giữ nút `‹`/`›`, hoặc dùng `←`/`→` và `A`/`D` để duyệt nhanh.

Trong các chế độ Semantic, nhấn **Enter** trong ô truy vấn sẽ tìm câu gốc và
đồng thời yêu cầu gợi ý mở rộng; chọn một gợi ý để tìm lại. **Shift+Enter**
vẫn xuống dòng. Nút **Mở rộng (Enter)** cho phép lấy gợi ý thủ công.

Jina nhận trực tiếp cả tiếng Việt và tiếng Anh. Caption corpus hiện là tiếng Anh
nhưng nằm trong cùng không gian multilingual, vì vậy không cần dịch query trước.
Apple-CLIP có tùy chọn dịch sát nghĩa query sang tiếng Anh bằng Groq trước khi
encode; tùy chọn này bật mặc định và nằm ngay dưới ô truy vấn. Giao diện hiện
câu tiếng Anh đã dùng.

## 1. Cấu trúc project hoàn chỉnh

Sau khi clone code và chuẩn bị artifact, project có cấu trúc đầy đủ như sau.
Các mục `[GitHub]` được commit; các mục `[Artifact]` tải từ Drive/Hugging Face và
được `.gitignore` chặn; các mục `[Generated]` được tạo trên máy lúc setup.

```text
AIC2026/
├── app.py                                      # [GitHub] Flask backend/API
├── retrieval_data.py                           # [GitHub] load metadata OCR/
├── semantic_search.py                          # [GitHub] Jina/Apple-CLIP encoder và NPY search
├── index.html                                  # [GitHub] giao diện
├── script.js                                   # [GitHub] logic frontend
├── style.css                                   # [GitHub] CSS
├── logo_wud.jpg                                # [GitHub] ảnh giao diện
│
├── requirements.txt                            # [GitHub] toàn bộ dependencies
├── .gitignore                                  # [GitHub]
├── artifacts-manifest.json                     # [GitHub] path/shape/count artifact
├── README.md                                   # [GitHub]
│
├── scripts/                                    # [GitHub]
│   ├── prepare_data.py                         # tải, unzip và validate artifact
│   └── build_filtered_ocr_metadata.py          # tái tạo OCR metadata đã lọc
│
├── keyframes/                                  # [Artifact]
│   ├── L21/L21_V001/000000.webp
│   ├── L22/...
│   └── L30/...
│
├── embedding/jina/                             # [Artifact]
│   ├── jina_embeddings_npy/
│   │   ├── L21.npy
│   │   ├── ...
│   │   └── L30.npy
│   └── caption_embeddings_npy/
│       ├── L21.npy
│       ├── ...
│       └── L30.npy
│
├── embedding/apple_finetuned/                  # [Artifact]
│   ├── apple_clip_epoch_5_inference.pt          # checkpoint không chứa optimizer
│   ├── L21-*.zip ... L30-*.zip                  # giữ ZIP; hoặc giải nén thành:
│   ├── L21/shard_000000.npz ...
│   └── L30/shard_*.npz
│
├── ocr/                                        # [Artifact]
│   ├── metadata_ocr_filtered.zip               # file tải về
│   └── metadata_ocr_filtered/                  # folder sau khi unzip
│       └── metadata/
│           ├── L21_V001.json
│           ├── ...
│           └── L30_*.json
│
├── asr/metadata_asr_clean/                     # [Artifact]
│   ├── L21_V001.json
│   ├── ...
│   └── L30_*.json
│
├── yolov8n.pt                                  # [Artifact, optional] Auto-Crop
└── .cache/                                     # [Generated]
    ├── huggingface/                            # pretrained Jina cache
    └── apple_clip_vectors/L21.npy ... L30.npy  # mmap cache dựng từ ZIP
```

Các nguồn dùng để **tạo lại artifact**, không cần trên máy người dùng cuối:

```text
Captions/                                       # caption CSV thô
OCR_original_no_LLM/OCR/L21.jsonl ... L30.jsonl
ocr/metadata_ocr/                               # metadata canonical chưa nhúng OCR
embedding/jina/encode_captions.py                # encoder caption offline
embedding/clip/mapping (1).json                  # mapping chỉ dùng lúc encode offline
```

Nếu muốn để data ngoài repository, giữ nguyên cây con artifact trong một folder
khác, ví dụ `D:\AIC2026-data`, rồi cấu hình các biến đường dẫn như phần chạy local.

## 2. Phân chia code và artifact

Nên dùng cả hai lớp sau:

1. **GitHub** lưu code, `requirements.txt`, manifest và tài liệu.
2. **Hugging Face Dataset private hoặc Google Drive** lưu keyframe, vector và metadata nặng.

Không đưa dữ liệu vài chục GB vào Git history. Người dùng chỉ cần clone code,
tải artifact đúng layout, tạo Python virtual environment và cài `requirements.txt`.

Lưu ý khi chia sẻ/triển khai: checkpoint Jina này công bố theo giấy phép
`CC-BY-NC-4.0`. Hãy kiểm tra lại điều khoản nếu mục đích sử dụng có yếu tố thương mại.

## 3. Cấu trúc artifact bắt buộc

Chuẩn bị một data root theo đúng layout này:

```text
data-root/
├── keyframes/
│   ├── L21/L21_V001/000000.jpg
│   └── ...
├── embedding/jina/
│   ├── jina_embeddings_npy/L21.npy ... L30.npy
│   └── caption_embeddings_npy/L21.npy ... L30.npy
├── embedding/apple_finetuned/
│   ├── apple_clip_epoch_5_inference.pt
│   └── L21-*.zip ... L30-*.zip              # hoặc L21/...L30/shard_*.npz
├── ocr/
│   ├── metadata_ocr_filtered.zip           # File vận chuyển/tải về
│   └── metadata_ocr_filtered/              # Runtime dùng folder đã giải nén
│       └── metadata/*.json
├── asr/metadata_asr_clean/
│   └── *.json
└── yolov8n.pt                    # tùy chọn, chỉ cho Auto-Crop
```

Các bộ vector phải có cùng thứ tự row với metadata, 1024 chiều,
đã L2-normalize. Tổng cộng phải có 317.961 rows. Số row từng collection nằm trong
`artifacts-manifest.json`. Apple-CLIP export dùng `float16` trong NPZ; lần khởi
động đầu app xác minh toàn bộ `image_names`, chuyển sang cache mmap `float32` và
những lần sau dùng lại cache đó. Có thể giữ nguyên 10 ZIP hoặc giải nén thành các
folder `L21`…`L30`; nếu tồn tại cả hai, runtime ưu tiên folder đã giải nén.

`metadata_ocr_filtered.zip` chứa cả metadata canonical và `ocr_text` lấy từ OCR
original sau khi lọc ticker L21/L22. Trước khi chạy, giải nén ZIP vào
`ocr/metadata_ocr_filtered/`; `prepare_data.py` tự làm bước này nếu folder
chưa có. Runtime không còn cần mang theo
`metadata_ocr/` cũ hoặc `OCR_original_no_LLM/`. Hai nguồn đó chỉ cần giữ ở máy
tạo artifact nếu muốn chạy lại `scripts/build_filtered_ocr_metadata.py`.

Giải nén thủ công trên PowerShell:

```powershell
New-Item -ItemType Directory -Force D:\AIC2026-data\ocr\metadata_ocr_filtered
Expand-Archive `
  D:\AIC2026-data\ocr\metadata_ocr_filtered.zip `
  D:\AIC2026-data\ocr\metadata_ocr_filtered `
  -Force
```

### Tải từ Hugging Face Dataset

Đặt nguyên layout trên trong một dataset repo, sau đó:

```powershell
python scripts/prepare_data.py `
  --repo-id YOUR_USER/aic2026-retrieval-artifacts `
  --data-dir D:\AIC2026-data
```

Repo private cần đặt `$env:HF_TOKEN`. Script tải snapshot, tự bung mọi file ZIP
trong `archives/`, rồi kiểm tra metadata và cả hai bộ vector.

### Tải từ Google Drive

Nén artifact theo collection để dễ resume, ví dụ `keyframes_L21.zip`, và để mỗi
ZIP chứa luôn đường dẫn đích như `keyframes/L21/...`. Tải các ZIP về
`D:\AIC2026-data\archives`, rồi chạy:

```powershell
python scripts/prepare_data.py --data-dir D:\AIC2026-data
```

Muốn kiểm tra tồn tại đủ từng ảnh (chậm hơn):

```powershell
python scripts/prepare_data.py --data-dir D:\AIC2026-data --full
```

## 4. Chạy local trên Windows

Yêu cầu Python 3.10/3.11 và NVIDIA GPU được khuyến nghị. Tạo môi trường:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu artifact nằm ngay trong folder project như máy gốc, không cần cấu hình path.
Nếu artifact nằm ở `D:\AIC2026-data`, đặt biến môi trường:

```powershell
$dataRoot = "D:\AIC2026-data"
$env:AIC_KEYFRAMES_DIR = "$dataRoot\keyframes"
$env:AIC_OCR_METADATA_PATH = "$dataRoot\ocr\metadata_ocr_filtered"
$env:AIC_ASR_METADATA_DIR = "$dataRoot\asr\metadata_asr_clean"
$env:AIC_JINA_VECTORS_DIR = "$dataRoot\embedding\jina\jina_embeddings_npy"
$env:AIC_JINA_CAPTION_VECTORS_DIR = "$dataRoot\embedding\jina\caption_embeddings_npy"
$env:AIC_APPLE_CLIP_ARTIFACTS_DIR = "$dataRoot\embedding\apple_finetuned"
$env:AIC_APPLE_CLIP_CHECKPOINT_PATH = "$dataRoot\embedding\apple_finetuned\apple_clip_epoch_5_inference.pt"
$env:AIC_APPLE_CLIP_CACHE_DIR = "D:\AIC2026-cache\apple_clip_vectors"
$env:AIC_YOLO_MODEL_PATH = "$dataRoot\yolov8n.pt"
$env:AIC_CACHE_DIR = "D:\AIC2026-cache\huggingface"
python app.py
```

Mở `http://localhost:5000`. Jina và Apple-CLIP đều lazy-load ở truy vấn tương ứng.
Jina cần Internet ở lần tải pretrained đầu tiên; Apple-CLIP dùng checkpoint local.
Lần khởi động đầu sẽ dựng Apple-CLIP mmap cache từ các ZIP. `GROQ_API_KEY` dùng
cho Query Expansion và dịch query Apple-CLIP; nếu thiếu key, Apple-CLIP vẫn chạy
với query gốc và giao diện sẽ báo rõ chưa dịch.

Player ưu tiên file video trong `video/`, tìm đệ quy theo tên nội bộ như
`L21_V001.mp4`. Backend stream file với HTTP Range để tua trực tiếp trên trình
duyệt. Nếu không có file local tương ứng, hệ thống mới dùng URL YouTube trong
metadata.

Demo giao thông N081–N100 nằm ở mode **Giao thông**. Có thể nhập query tiếng Việt
như `xe máy màu đỏ`, `xe buýt màu trắng` hoặc `đường đông có ô tô và xe tải`.
Loại phương tiện được lọc bằng detection; màu sắc và ngữ cảnh được xếp hạng bằng
caption embedding. Timestamp được đọc từ `map-keyframes`, sau đó player mở video
local theo ID dạng `N081-V001`.

Nộp trực tiếp vòng chung kết dùng DRES v2. Trước giờ thi, xác nhận địa chỉ DRES
do BTC cung cấp; cấu hình bằng biến môi trường `BTC_API_BASE_URL` nếu khác mặc
định `https://eventretrieval.oj.io.vn`. Ví dụ trong PowerShell, trước khi chạy app:

```powershell
$env:BTC_API_BASE_URL = "https://DRES_HOST_DO_BTC_CUNG_CAP"
```

Không đưa `sessionID` vào code hoặc Git. Nhập sessionID vào ô được che trên UI,
nhập evaluationID rồi bấm **Kiểm tra DRES**. Nút này chỉ gọi API đọc danh sách
evaluation; nếu chỉ có một evaluation ACTIVE, UI tự điền ID. Không có bài nộp
nào được gửi khi kiểm tra. SessionID không được lưu vào `localStorage`.

Kiểm tra nhanh dịch vụ:

```powershell
Invoke-RestMethod http://localhost:5000/health
```

## 5. Biến môi trường

| Biến | Mặc định |
|---|---|
| `AIC_KEYFRAMES_DIR` | `keyframes` |
| `AIC_OCR_METADATA_PATH` | ưu tiên folder `ocr/metadata_ocr_filtered` |
| `AIC_OCR_TEXT_DIR` | tùy chọn; chỉ overlay khi dùng metadata legacy |
| `AIC_ASR_METADATA_DIR` | `asr/metadata_asr_clean` |
| `AIC_JINA_VECTORS_DIR` | `embedding/jina/jina_embeddings_npy` |
| `AIC_JINA_CAPTION_VECTORS_DIR` | `embedding/jina/caption_embeddings_npy` |
| `AIC_APPLE_CLIP_ARTIFACTS_DIR` | `embedding/apple_finetuned` |
| `AIC_APPLE_CLIP_CHECKPOINT_PATH` | `embedding/apple_finetuned/apple_clip_epoch_5_inference.pt` |
| `AIC_APPLE_CLIP_CACHE_DIR` | `.cache/apple_clip_vectors` |
| `AIC_YOLO_MODEL_PATH` | `yolov8n.pt` |
| `AIC_TRAFFIC_CAPTION_DIR` | `captionbatch2_emb/Video_N081-N100` |
| `AIC_TRAFFIC_DETECTION_PATH` | `detection segmentation/detection segmentation/Video_N081-N100_metadata.parquet` |
| `AIC_TRAFFIC_KEYFRAMES_DIR` | `keyframes_batch2/N081-N100/keyframes` |
| `AIC_TRAFFIC_MAP_DIR` | `keyframes_batch2/N081-N100/map-keyframes` |
| `AIC_CACHE_DIR` | `.cache/huggingface` |
| `GROQ_API_KEY` | rỗng; Query Expansion và dịch Apple-CLIP bị tắt |
| `BTC_API_BASE_URL` | `https://eventretrieval.oj.io.vn`; cần xác nhận host thật với BTC |

## 6. Đưa cập nhật UI/DRES lên GitHub

Không dùng `git add .` trong workspace chứa artifact lớn. Với bản cập nhật này,
chỉ đưa đúng các file code/tài liệu đã thay đổi vào commit. Chạy trong PowerShell
tại thư mục repo:

```powershell
Set-Location -LiteralPath 'D:\AIC2026'
git fetch origin
git status --short --branch
git add -- README.md app.py dres_gateway.py index.html script.js style.css `
  submission-builder.css submission-builder.html submission-builder.js `
  tests/submission_builder_smoke.js tests/test_dres_gateway.py
git diff --cached --stat
git diff --cached --check
git commit -m "Improve final-round submission UI and DRES handling"
git push origin main
```

Trước `git commit`, có thể dùng `git diff --cached --name-only` để kiểm tra danh
sách file. Nếu `git status` báo `behind` sau khi fetch, cần đồng bộ thay đổi từ
remote trước khi push; không dùng `git push --force`. Nếu cần đẩy lên repo của
người khác, tài khoản phải có quyền ghi; nếu không, tạo fork/PR theo quy trình
của nhóm. Tuyệt đối không commit `sessionID`, `GROQ_API_KEY` hoặc file `.env`.

Không push các folder/file sau: `keyframes/`, `embedding/`, `ocr/`,
`OCR_original_no_LLM/`, `asr/`, `Captions/`, `.cache/`, `.venv/`, `*.npy`,
`*.zip` và model weights. `.gitignore` đã chặn các nhóm này.

## 7. API retrieval chính

| Endpoint | Nội dung |
|---|---|
| `GET /health` | trạng thái artifact/runtime |
| `GET /semantic_models` | trạng thái Apple-CLIP, Jina và Jina Hybrid |
| `POST /search` | `semantic_model`: `apple-clip`, `jina` hoặc `jina-hybrid` |
| `POST /search_ocr` | OCR BM25 |
| `POST /search_asr` | ASR BM25 |
| `POST /search_fusion` | `query_jina`, `query_ocr`, `query_asr` + weights |
| `POST /search_similar_image` | một ảnh multipart |
| `POST /search_trake_02` | mảng `events` theo thứ tự |
| `POST /search_trake_image` | ít nhất hai ảnh multipart |
| `POST /get_keyframe_map` | bản đồ thời gian, `frame_idx`, đường dẫn keyframe và FPS theo video |
| `POST /submission/resolve_candidates` | map keyframe sang `frame_idx` thật |
| `POST /submission/neighbors` | lấy frame cùng video trong biên thời gian quanh các mốc ghim |
| `POST /submission/playback` | tìm timestamp video gần `frame_idx` để kiểm tra |
| `POST /submission/export` | validate và tạo `submission.zip` |
| `POST /dres/status` | kiểm tra session và evaluation ACTIVE qua DRES; không gửi bài |
| `POST /submit_answer` | gửi một đáp án DRES v2 và trả trạng thái đã nhận/từ chối/chưa rõ |

## 8. Tạo file nộp vòng sơ tuyển AIC26

Sau khi chạy app, mở `http://localhost:5000/submission-builder` hoặc bấm
**📦 Bài nộp** trên header trang search.

1. Tạo/import các query có tên kết thúc bằng `-kis`, `-qa` hoặc `-trake`.
2. Chọn query đang làm trên header trang search.
3. Mở video YouTube để quan sát và tua/phát tới vị trí cần tìm. Dải keyframe
   đi theo thời gian phát; bấm **📌 Ghim timestamp đang phát** để ghim keyframe
   gần nhất trên bản đồ metadata của video. Ghim `frame_idx` của keyframe local,
   **không** lấy thời gian YouTube nhân FPS để tạo `frame_idx`. Với TRAKE, ghim
   cả chuỗi từ kết quả TRAKE thay vì ghim một frame đơn.
4. Trong Submission Builder, kéo tay cầm `⠿` để sắp xếp các dòng ghim, chọn biên thời gian rồi bấm **Fill quanh các frame
   ghim**, hoặc dùng **◎ Fill quanh** tại một dòng cụ thể. Auto-fill chỉ lấy frame
   cùng video trong khoảng thời gian đó, không dùng top-K image search.
5. Nếu đã biết đáp án, nhập thẳng `video_id,frame_idx` vào ô **Thêm kết quả thủ
   công**. TRAKE nhập `video_id` rồi toàn bộ `frame_idx` theo thứ tự event.
6. Bấm thumbnail để mở thẻ metadata/video ngay trong Builder; dùng **▶ Xem**
   nếu muốn mở video tại timestamp gần frame đã chọn trong tab riêng.
7. Bấm **Tải submission.zip**. Backend kiểm tra `frame_idx`, số event TRAKE,
   answer QA và tạo đúng cấu trúc `submission/*.csv` không có header.

Khi dùng bảng **Nộp Đáp Án** để gửi trực tiếp, YouTube vẫn phục vụ xem/tua video
và định vị keyframe lân cận. Mốc dùng để nộp lấy từ keyframe gần nhất trong
metadata local: **KIS** dùng `pts_time` của hai lần **Click (Set Start/End)**;
**QA** dùng `pts_time` của keyframe tại lúc gửi; **TRAKE** dùng `frame_idx` của
keyframe khi bấm **Click (Add Frame)**. Có thể nhập thời gian KIS hoặc
`frame_idx` TRAKE thủ công khi cần. Thời gian local ưu tiên `pts_time` trong
metadata; chỉ dùng `frame_idx / fps` khi thiếu mốc thời gian hợp lệ. Nếu bản đồ
keyframe chưa tải được, thao tác chọn keyframe để nộp sẽ báo lỗi thay vì suy
`frame_idx` từ đồng hồ YouTube. Ở phía giao diện, KIS chấp nhận `Start = End`
cho một mốc thời gian duy nhất; chỉ từ chối khi `End < Start`. Cần xác nhận
máy chủ DRES của BTC chấp nhận khoảng thời gian bằng 0 trước khi dùng trong thi.

Trước khi nộp trực tiếp, thẻ bên phải hiện video ID, nội dung đáp án và thumbnail
keyframe local để kiểm tra nhanh. Bấm **Nộp đáp án** hoặc **Ctrl+Enter** khi đang
đặt con trỏ trong bảng nộp; hệ thống khóa nút trong lúc gửi và không tự gửi lại
khi mất kết nối. Phản hồi **DRES đã nhận** khác với phán quyết **ĐÚNG/SAI**:
HTTP 202 có thể chưa có phán quyết. Nếu
hiện **Chưa rõ DRES đã nhận**, kiểm tra trên DRES trước khi thử lại cùng đáp án.
UI cảnh báo khi gửi lại cùng payload trong cùng phiên làm việc.

Việc chọn keyframe hiện dựa trên mốc thời gian gần nhất trong bản đồ metadata,
không so khớp nội dung hình ảnh. Nếu bản YouTube và video local bị lệch timeline,
hãy đối chiếu ảnh keyframe và thời gian local trước khi gửi đáp án.

Có thể dùng **Merge nguyên folder submission CSV** trong mục **Dự phòng** để nhập
trực tiếp một folder chứa các file `*-kis.csv`, `*-qa.csv`, `*-trake.csv`. Các dòng
được đưa vào phần ghim theo thứ tự, dữ liệu local được ưu tiên và app tải backup JSON
trước khi merge.

Bản nháp được lưu trong `localStorage` của trình duyệt. Dùng **Xuất project
JSON** để sao lưu hoặc gửi cho teammate. Máy tổng hợp dùng **Merge project JSON
từ teammate**: app tự tải backup bản local trước, giữ thứ tự ghim local ở đầu,
nối các lựa chọn của teammate sau và loại dòng trùng. Nếu prompt/answer hoặc số
event xung đột, dữ liệu local được giữ và app báo số conflict. Luồng nộp trực
tiếp DRES độc lập với công cụ vòng sơ tuyển này.

## 9. Lỗi thường gặp

- **Jina Hybrid bị khóa**: thiếu hoặc sai một shard caption `L21.npy…L30.npy`;
  chạy lại `scripts/prepare_data.py` để thấy file/shape sai.
- **Apple-CLIP bị khóa**: kiểm tra đủ checkpoint inference và 10 ZIP `L21…L30`;
  xóa `.cache/apple_clip_vectors` rồi khởi động lại nếu artifact đã được thay mới.
- **Model không tải được**: kiểm tra Internet, `HF_TOKEN` nếu cache/repo private,
  và quyền ghi `AIC_CACHE_DIR`.
- **CUDA unavailable**: kiểm tra NVIDIA driver và chạy
  `python -c "import torch; print(torch.cuda.is_available())"`.
- **Auto-Crop tắt**: đặt đúng `yolov8n.pt`; các chế độ Jina vẫn chạy bình thường.
- **Out of memory**: đóng process Python khác đang chiếm VRAM rồi chạy lại app.
