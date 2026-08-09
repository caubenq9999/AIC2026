# MFusion-VR — Hệ thống truy vấn video/hình ảnh (AIC2025/2026)

Hệ thống event/video retrieval kiểu VBS (Video Browser Showdown): tìm keyframe trong kho video lớn bằng text (semantic), OCR, ASR, ảnh mẫu (similar search), hoặc chuỗi sự kiện (TRAKE), rồi nộp đáp án (QA/KIS/TRAKE) trực tiếp lên server chấm điểm của BTC.

## Kiến trúc

```
Browser (index.html/script.js/style.css)
        │  fetch() → JSON / multipart
        ▼
Flask app.py (root)
        ├─ CLIP (apple/DFN5B-CLIP-ViT-H-14-378 + LoRA "fine_tuned_model_lora_2025")
        │     → text/ảnh → vector 1024-d → FAISS IndexFlatIP (382k keyframes)
        ├─ BEIT3 (timm beit3_base_patch16_224.in22k_ft_in1k)
        │     → vector 768-d, dùng fusion re-rank cho Similar Search
        ├─ YOLOv8n (yolov8n.pt) → auto-crop vật thể trước khi encode CLIP (Similar Search)
        ├─ BLIP-ITM reranker (test_reranker.py) → rerank top-N (đang tắt: ENABLE_RERANKER=False)
        ├─ BM25 tự implement (class BM25 trong app.py, không cần service ngoài)
        │     ├─ bm25_ocr_index (từ ocr.json)   → /search_ocr
        │     └─ bm25_asr_index (từ asr_result/) → /search_asr
        ├─ Groq (model openai/gpt-oss-120b) → dịch VI→EN, Query Expansion
        └─ BTC_API_BASE_URL (eventretrieval.oj.io.vn) → /submit_answer forward đáp án
```

`app.py` là server **đang chạy thật** (production). Thư mục [code/](code/) chứa một bản UI/backend đơn giản hơn, cũ hơn — chỉ giữ để tham khảo, **không phải bản đang dùng**.

## Các chế độ tìm kiếm (search-mode)

| Mode | Endpoint | Input | Ghi chú |
|---|---|---|---|
| Semantic | `POST /search` | text | CLIP text→image; hỗ trợ dịch VI→EN, tìm trực tiếp theo Video ID |
| Query Expansion | `POST /expand_query` | text | Sinh 3 biến thể câu query bằng Groq, KHÔNG tự search — trả về để người dùng chọn 1 cái rồi search bằng `/search` |
| OCR | `POST /search_ocr` | text | BM25 tự implement (`bm25_ocr_index`), không cần service ngoài |
| ASR | `POST /search_asr` | text | BM25 tự implement (`bm25_asr_index`) + ghép đoạn liền kề |
| Fusion | `POST /search_fusion` | 3 ô text riêng (CLIP/OCR/ASR) + tỷ lệ trọng số | Gộp CLIP + OCR + ASR bằng Reciprocal Rank Fusion có trọng số |
| Similar Search | `POST /search_similar_image` | 1 ảnh upload | CLIP image search, auto-crop YOLO tuỳ chọn, **fusion CLIP↔BEIT3** qua slider |
| TRAKE | `POST /search_trake_02` | `events: ["sự kiện 1", "sự kiện 2", ...]` (mỗi sự kiện 1 ô nhập riêng, thêm/xóa được; vẫn nhận `query` ngăn bằng `;` khi gọi API trực tiếp) | **Temporal alignment**: tìm video chứa cả chuỗi sự kiện, rồi DP căn mỗi sự kiện về đúng 1 keyframe theo thứ tự thời gian tăng dần. Trả về mỗi video 1 chuỗi N frame |
| TRAKE Image | `POST /search_trake_image` | ≥2 ảnh upload | Tương tự TRAKE.02 nhưng theo ảnh mẫu |

Các endpoint hỗ trợ `group=true` để nhóm kết quả theo video (dạng "album", tiện cho TRAKE).

Nộp đáp án: `POST /submit_answer` (JSON `{evaluation_id, session_id, answer_payload}`) forward tới `BTC_API_BASE_URL`.

## Dữ liệu

| Thư mục/File | Nội dung |
|---|---|
| `keyframes/`, `Keyframes_K*/L*` | Ảnh keyframe trích từ video, đặt tên `K10_V001/001.jpg` |
| `map-keyframes/` | CSV map `n` (số frame) ↔ `frame_idx`, `pts_time`, `fps` |
| `media-info/` | JSON metadata mỗi video (trong đó có `watch_url` YouTube) |
| `ocr.json`, `asr_result/` | Dữ liệu OCR/ASR thô, được tải thẳng vào bộ nhớ lúc khởi động để build `bm25_ocr_index`/`bm25_asr_index` |
| `image_embeddings.npy` + `image_paths.json` | Vector CLIP đầy đủ — 382,299 keyframe × 1024-d (float16) |
| `beit3_vectors/vectors_beit3/` | Vector BEIT3 theo shard (mỗi batch K*/L* một file `*_beit3_embeddings.npy` + `*_beit3_paths.json`), 768-d |
| `vectors/` | Shard CLIP cũ, không đầy đủ (chỉ K10–K20, L25–L30) — đã bị thay thế bởi `image_embeddings.npy`, giữ lại để tham khảo |
| `fine_tuned_model_lora_2025/` | Adapter LoRA fine-tune trên CLIP |

> BEIT3 dùng ở đây (`timm/beit3_base_patch16_224.in22k_ft_in1k`) chỉ có vision tower (checkpoint phân loại ImageNet), **không có text tower** — nên chỉ dùng được cho so ảnh-với-ảnh (Similar Search), không dùng cho tìm kiếm bằng text.

## Chạy hệ thống

Cần Python có đủ gói trong `requirements.txt` (bao gồm `torch`+CUDA nếu muốn chạy GPU, `ultralytics`, `peft`, `timm`...) và API key Groq (biến `GROQ_API_KEY` ở đầu [app.py](app.py) — **hiện đang hardcode tạm để test**, cần thay bằng key thật trước khi build/commit). Không còn phụ thuộc Elasticsearch — OCR/ASR chạy bằng BM25 tự implement, build thẳng từ `ocr.json`/`asr_result/` lúc khởi động (~12s).

```bash
pip install -r requirements.txt
python app.py            # chạy ở http://localhost:5000
```

## Tính năng mới gần đây

- **TRAKE viết lại theo Temporal Alignment** — bản cũ tìm frame mà *tất cả* các phần mô tả cùng xuất hiện trong cửa sổ hẹp ±N frame, sai bản chất TRAKE vì các sự kiện trong một chuỗi (chạy đà → giậm nhảy → bay qua xà → tiếp đất) nằm rải rác cách nhau hàng trăm frame. Bản mới chạy đúng 2 giai đoạn của BTC: (1) *Retrieval* — gộp điểm mọi sự kiện theo từng video để tìm video chứa cả chuỗi; (2) *Alignment* — quy hoạch động (`_align_event_sequence`) chọn cho mỗi sự kiện đúng 1 keyframe sao cho thứ tự thời gian tăng dần và tổng điểm CLIP lớn nhất. DP còn **phạt khoảng cách thời gian**: 2 sự kiện liên tiếp cách nhau quá ngưỡng (ô "Cách nhau tối đa", mặc định 30 giây) bị trừ `0.01 điểm/giây` vượt ngưỡng — vượt 30s là mất trọn giá trị một sự kiện khớp. Không ràng buộc này thì DP hay ghép các sự kiện cách nhau vài phút, vốn là những cảnh không liên quan trong cùng video (keyframe cách nhau ~2.8s, video dài ~16 phút). Phạt mềm chứ không chặn cứng, phòng khi đáp án thật hơi thưa.

Kết quả hiện dạng "chuỗi": mỗi video một hàng N ảnh theo thứ tự sự kiện, mỗi ô ghi khoảng cách tới sự kiện liền trước (tô đỏ khi vượt ngưỡng), header ghi tổng thời lượng chuỗi, kèm nút **"⬇ Dùng chuỗi này để nộp"** đổ thẳng cả N `frame_idx` vào ô nộp TRAKE (trước đây phải click từng frame một). Phần nhập cũng đổi từ một ô text ngăn bằng `;` thành **danh sách ô riêng cho từng sự kiện**, có nút "+ Thêm sự kiện" và nút ✕ xóa từng ô (khoá khi chỉ còn 2 ô — dưới 2 thì không còn là chuỗi).
- **Fusion Search (CLIP + OCR + ASR)** — 3 ô nhập riêng cho từng nguồn + 3 thanh trượt tỷ lệ trọng số (mặc định 1/3 mỗi nguồn), gộp bằng Reciprocal Rank Fusion có trọng số. Mỗi kết quả có `matched_by` cho biết khớp bởi nguồn nào.
- **Fusion CLIP ↔ BEIT3 (Similar Search)** — thanh trượt trong chế độ "Tìm Ảnh (Tương tự)" điều chỉnh trọng số giữa CLIP (ngữ nghĩa) và BEIT3 (chi tiết/fine-grained) khi rerank kết quả tìm ảnh bằng ảnh mẫu.
- **Query Expansion** — nút "🔎 Mở rộng câu truy vấn" ở chế độ Semantic Search: dùng Groq sinh 3 biến thể câu query (sát nghĩa / nhấn vai trò / nhấn nơi chốn, theo `[AIC2026] - Query expansion.docx`), hiện ra để người dùng bấm chọn 1 biến thể rồi search luôn bằng biến thể đó (không tự động gộp cả 3 nữa — tránh tình trạng operator không biết hệ thống đã tìm bằng câu gì).
- **UI**: banner lớn được thu gọn thành thanh tiêu đề mỏng; select box loại câu hỏi (QA/KIS/TRAKE) chuyển ra thanh luôn hiển thị ở đầu trang thay vì phải bấm vào frame rồi cuộn xuống mới chọn được (theo góp ý trong `[AIC2026] - Đề xuất UI/DE_XUAT.jpg`).

## Script phụ trợ (root)

- `elasticocr.py` / `elasticasr.py` — **không còn dùng** (còn lại từ thời còn Elasticsearch, `app.py` giờ chạy BM25 tự implement, không cần chạy 2 file này nữa).
- `demo_ocrsearch.py` — script thử nghiệm query cũ, tham chiếu tới Elasticsearch, không còn khớp với `app.py` hiện tại.
- `test_reranker.py` — `BlipReranker` (BLIP-ITM) dùng để rerank kết quả (hiện đang tắt trong `app.py`).
- `playvideo.py` — phát thử video cục bộ bằng VLC.

## Tài liệu thiết kế

- `[AIC2026] - Query expansion.docx` — ý tưởng & prompt cho Query Expansion (PLAN A).
- `[AIC2026] - Đề xuất UI/` — ảnh chụp UI hiện tại (`UI_present.jpg`), đề xuất cải tiến (`DE_XUAT.jpg`), và ảnh tham khảo UI của các đội khác tại VBS (`ref1-3.jpg`).
