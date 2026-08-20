# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import os
from pathlib import Path

# Mọi đường dẫn mặc định đều neo theo vị trí app.py, không phụ thuộc cwd.
BASE_DIR = Path(__file__).resolve().parent


def project_path(environment_variable, *relative_parts):
    configured_path = os.getenv(environment_variable)
    if configured_path:
        configured_path = Path(configured_path).expanduser()
        if not configured_path.is_absolute():
            configured_path = BASE_DIR / configured_path
        return configured_path.resolve()
    return BASE_DIR.joinpath(*relative_parts).resolve()


def resolve_ocr_metadata_path():
    """Resolve OCR metadata from an override, directory, or legacy ZIP archive."""
    configured_path = os.getenv("AIC_OCR_METADATA_PATH", "").strip()
    if configured_path:
        return project_path("AIC_OCR_METADATA_PATH")

    candidates = (
        BASE_DIR / "ocr" / "metadata_ocr_filtered",
        BASE_DIR / "ocr" / "metadata_ocr_filtered.zip",
        BASE_DIR / "ocr" / "metadata_ocr",
        BASE_DIR / "ocr" / "metadata_ocr.zip",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Keep the error emitted by load_retrieval_data deterministic and useful.
    return candidates[0].resolve()


# Có thể đổi vị trí cache bằng AIC_CACHE_DIR mà không cần sửa source.
CACHE_DIR = project_path("AIC_CACHE_DIR", ".cache", "huggingface")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["TORCH_HOME"] = str(CACHE_DIR)
os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR)

#BTC_EVALUATION_ID = "dda49193-bcb6-4e7d-880f-bf7ec60046ee"
#BTC_SESSION_ID = "tlMIiLdLV-yTB_ENJx6gDtimFMNYL5qk"
BTC_API_BASE_URL = "https://eventretrieval.oj.io.vn"
import torch
import requests
import numpy as np
import json
import csv
import zipfile
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_from_directory, send_file, g, abort
import gc
from retrieval_data import (
    load_asr_metadata,
    load_retrieval_data,
    overlay_ocr_jsonl,
    parse_keyframe_path,
)
from semantic_search import (
    JinaTextEncoder,
    ModelUnavailableError,
    ShardedNpyIndex,
)
from groq import Groq
from flask_cors import CORS
# from rank_bm25 import BM25Okapi # <-- XÓA BỎ (Không dùng thư viện nữa)
import re 
import collections
import math 
import bisect 
from PIL import Image # (THÊM MỚI) Thêm PIL để xử lý ảnh upload
import io # (THÊM MỚI) Thêm io
from ultralytics import YOLO # (THÊM MỚI) YOLOv8 cho auto-crop pre-processing


print("--- KHỞI ĐỘNG HỆ THỐNG TRUY VẤN HÌNH ẢNH ---")

# --- 2. CẤU HÌNH ---
index_name = "aic_ocr_index"
KEYFRAMES_DIR = project_path("AIC_KEYFRAMES_DIR", "keyframes")
OCR_METADATA_PATH = resolve_ocr_metadata_path()
OCR_TEXT_DIR = project_path(
    "AIC_OCR_TEXT_DIR", "OCR_original_no_LLM", "OCR"
)
ASR_METADATA_DIR = project_path("AIC_ASR_METADATA_DIR", "asr", "metadata_asr_clean")
YOLO_MODEL_PATH = project_path("AIC_YOLO_MODEL_PATH", "yolov8n.pt")
JINA_VECTORS_DIR = project_path(
    "AIC_JINA_VECTORS_DIR", "embedding", "jina", "jina_embeddings_npy"
)
JINA_CAPTION_VECTORS_DIR = project_path(
    "AIC_JINA_CAPTION_VECTORS_DIR",
    "embedding",
    "jina",
    "caption_embeddings_npy",
)

# --- 3. CLASS BM25 TỰ IMPLEMENT CỦA BẠN ---
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.doc_count = len(corpus)
        self.doc_freqs = self._calculate_doc_freqs()
        self.idf = self._calculate_idf()
    def _calculate_doc_freqs(self):
        doc_freqs = {}
        for doc in self.corpus:
            for term in set(doc):
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        return doc_freqs
    def _calculate_idf(self):
        idf = {}
        for term, freq in self.doc_freqs.items():
            idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
        return idf
    def get_scores(self, query):
        # Giữ nguyên scoring BM25 cũ cho các nhánh khác (đặc biệt ASR).
        scores = np.zeros(self.doc_count)
        for term in query:
            if term not in self.idf:
                continue
            term_freqs = np.fromiter(
                (doc.count(term) for doc in self.corpus),
                dtype=np.float64,
                count=self.doc_count,
            )
            numerator = term_freqs * (self.k1 + 1)
            denominator = term_freqs + self.k1 * (
                1 - self.b + self.b * (np.array(self.doc_len) / self.avgdl)
            )
            scores += self.idf[term] * (numerator / denominator)
        return scores

    def get_scores_with_match_counts(self, query):
        """Return BM25 scores and number of distinct query terms found per doc."""
        scores = np.zeros(self.doc_count)
        match_counts = np.zeros(self.doc_count, dtype=np.uint16)
        # Repeating a word in the user's query must not multiply its weight.
        distinct_query = list(dict.fromkeys(query))
        for term in distinct_query:
            if term not in self.idf:
                continue
            term_freqs = np.fromiter(
                (doc.count(term) for doc in self.corpus),
                dtype=np.float64,
                count=self.doc_count,
            )
            match_counts += term_freqs > 0
            numerator = term_freqs * (self.k1 + 1)
            denominator = term_freqs + self.k1 * (1 - self.b + self.b * (np.array(self.doc_len) / self.avgdl))
            scores += self.idf[term] * (numerator / denominator)
        return scores, match_counts
# --- KẾT THÚC CLASS BM25 ---

# --- 4. CẤU HÌNH GROQ API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = "openai/gpt-oss-120b"
try:
    if not GROQ_API_KEY:
        raise ValueError("Chưa cấu hình biến môi trường GROQ_API_KEY.")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Kết nối với Groq API thành công.")
except Exception as e:
    print(f"Lỗi khi cấu hình Groq API: {e}")
    groq_client = None

def groq_generate(prompt):
    """Gọi Groq chat completions (model GROQ_MODEL), trả về text output."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# --- 5. TẢI CÁC MODEL VÀ DỮ LIỆU ---
print("Đang tải model và các tài nguyên, vui lòng đợi...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# (THÊM MỚI) Tải YOLOv8n model cho auto-crop pre-processing
yolo_model = None
try:
    yolo_model = YOLO(str(YOLO_MODEL_PATH))  # Tự động tải về nếu chưa có
    yolo_model.to(device)
    print(f"Model YOLOv8n đã được tải thành công lên thiết bị: {device.upper()}")
except Exception as e:
    print(f"CẢNH BÁO: Không thể tải YOLOv8n model: {e}. Tính năng auto_crop sẽ bị vô hiệu hóa.")
    yolo_model = None

retrieval_data = load_retrieval_data(
    OCR_METADATA_PATH,
    KEYFRAMES_DIR,
)
image_records = retrieval_data.image_records
metadata_cache = retrieval_data.metadata_cache
keyframe_time_cache = retrieval_data.keyframe_time_cache
video_frame_ids = retrieval_data.video_frame_ids
video_url_cache = retrieval_data.video_url_cache
print(f"Loaded {len(image_records)} embedding/OCR records from {len(metadata_cache)} videos.")

# File filtered đã nhúng OCR text. Với metadata legacy, overlay JSONL vẫn được
# hỗ trợ để tái tạo đúng cùng kết quả mà không sửa nguồn canonical.
ocr_text_is_embedded = OCR_METADATA_PATH.name.lower() in {
    "metadata_ocr_filtered",
    "metadata_ocr_filtered.zip",
}
if OCR_TEXT_DIR.is_dir() and not ocr_text_is_embedded:
    ocr_overlay_stats = overlay_ocr_jsonl(OCR_TEXT_DIR, image_records)
    print(
        "Loaded OCR text from "
        f"{OCR_TEXT_DIR}: {ocr_overlay_stats['rows']:,} rows / "
        f"{ocr_overlay_stats['files']} shards "
        f"({ocr_overlay_stats['blank_texts']:,} blank; "
        f"filtered {ocr_overlay_stats['filtered_ticker_lines']:,} ticker lines in L21/L22)."
    )
else:
    if ocr_text_is_embedded:
        print(f"OCR text đã được nhúng và lọc trong {OCR_METADATA_PATH}.")
    else:
        print(
            f"CẢNH BÁO: Không tìm thấy OCR JSONL {OCR_TEXT_DIR}; "
            "OCR mode dùng ocr_text trong metadata canonical."
        )
ocr_data = image_records

# Hai index Jina dùng exact search trên NPY mmap và chung một thứ tự metadata.
jina_semantic_index = ShardedNpyIndex(
    "Jina",
    JINA_VECTORS_DIR,
    image_records,
    expected_dimension=1024,
)
jina_caption_index = None
jina_caption_index_reason = "Caption embeddings chưa được tạo."
try:
    expected_caption_shards = {f"L{number}.npy" for number in range(21, 31)}
    present_caption_shards = {
        path.name for path in JINA_CAPTION_VECTORS_DIR.glob("L*.npy")
    }
    missing_caption_shards = sorted(expected_caption_shards - present_caption_shards)
    if missing_caption_shards:
        raise FileNotFoundError(
            "Caption embeddings chưa đầy đủ; còn thiếu "
            + ", ".join(missing_caption_shards)
            + ". Chạy embedding/jina/encode_captions.py để tiếp tục."
        )
    jina_caption_index = ShardedNpyIndex(
        "Jina Caption",
        JINA_CAPTION_VECTORS_DIR,
        image_records,
        expected_dimension=1024,
    )
    jina_caption_index_reason = (
        f"Đã map {jina_caption_index.ntotal:,} caption vectors."
    )
except (FileNotFoundError, ValueError) as exc:
    # Jina image vẫn dùng độc lập được nếu caption artifact chưa đầy đủ.
    jina_caption_index_reason = str(exc)
    print(f"Caption search chưa sẵn sàng: {exc}")
jina_text_encoder = JinaTextEncoder(device=device)
print(f"Đã map Jina image: {jina_semantic_index.ntotal:,} vector, 1024 chiều.")

print("Loading ASR metadata...")
asr_data, asr_corpus_tokenized, asr_video_map = load_asr_metadata(ASR_METADATA_DIR)
print(f"Loaded {len(asr_data)} ASR segments from {len(asr_video_map)} videos.")

# --- 6. XÂY DỰNG CÁC INDEX TÌM KIẾM ---

# HÀM LÀM SẠCH OCR
def clean_ocr_text(text):
    text_lower = text.lower()
    patterns = [
        r'\d{1,2}:\d{2}(:\d{2})?', 
        r'\b(htv|htvt)\d?\b',   
        r'\b(website|fanpage|youtube|tintuc|www|fb\.com)\b',
        r'\.(com|vn)',           
        r'gưu họ c',           
        r'\bfiv\b'            
    ]
    for pattern in patterns:
        text_lower = re.sub(pattern, ' ', text_lower, flags=re.IGNORECASE)
    text_lower = re.sub(r'\b[a-zA-Z]\b', ' ', text_lower)
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    return text_lower


def tokenize_ocr_text(text):
    """Tokenize OCR consistently and detach punctuation from Vietnamese words."""
    return re.findall(r"[^\W_]+", clean_ocr_text(text), flags=re.UNICODE)


def tokenize_asr_text(text):
    """Tokenize ASR without applying OCR-specific logo/time cleanup rules."""
    return re.findall(r"[^\W_]+", str(text or "").lower(), flags=re.UNICODE)
# --- KẾT THÚC HÀM ---


# Index 2: BM25 cho OCR
print("Đang xây dựng index tìm kiếm với BM25 (cho OCR)...")
bm25_ocr_index = None
if ocr_data:
    tokenized_corpus_ocr = []
    print("Bắt đầu làm sạch dữ liệu OCR...")
    for item in ocr_data:
        original_text = item.get('ocr_text', '')
        tokenized_corpus_ocr.append(tokenize_ocr_text(original_text))
    print("Làm sạch OCR hoàn tất. Đang huấn luyện BM25...")
    # OCR của slide/bài giảng thường dài hơn caption/logo rất nhiều. b thấp
    # giúp BM25 không phạt độ dài quá tay; coverage/phrase bonus ở
    # ocr_candidates() đảm bảo khớp đủ cụm từ vẫn đứng trên khớp một từ ngắn.
    bm25_ocr_index = BM25(tokenized_corpus_ocr, k1=1.5, b=0.20)
    print(f"Xây dựng index BM25 (OCR) (tự implement) hoàn tất cho {len(tokenized_corpus_ocr)} văn bản.")
else:
    print("Không có dữ liệu OCR để xây dựng index BM25.")


# Index 3: BM25 cho ASR
print("Đang xây dựng index tìm kiếm với BM25 (cho ASR)...")
bm25_asr_index = None
if asr_data:
    # Dùng cùng tokenizer/ranking policy mới của OCR nhưng không chạy các regex
    # cleanup riêng cho logo, timestamp và website của OCR.
    asr_corpus_tokenized = [
        tokenize_asr_text(segment.get("text", "")) for segment in asr_data
    ]
    bm25_asr_index = BM25(asr_corpus_tokenized, k1=1.5, b=0.20)
    print(f"Xây dựng index BM25 (ASR) (tự implement) hoàn tất cho {len(asr_corpus_tokenized)} văn bản.")
else:
    print("Không có dữ liệu ASR để xây dựng index BM25.")


# --- 7. TẠO FLASK APP ---
app = Flask(__name__)
CORS(app, allow_headers="*")

# HÀM HELPER ĐỂ TÌM KEYFRAME GẦN NHẤT
def find_closest_keyframe(video_id, target_time):
    if video_id not in keyframe_time_cache:
        return {"frame_n": None, "frame_idx": None}
    cache_entry = keyframe_time_cache[video_id]
    times = cache_entry["times"]
    data = cache_entry["data"]
    if not times:
        return {"frame_n": None, "frame_idx": None}
    index = bisect.bisect_left(times, target_time)
    if index == 0:
        best_match_data = data[0]
    elif index == len(times):
        best_match_data = data[-1]
    else:
        time_before = times[index - 1]
        time_after = times[index]
        if (target_time - time_before) < (time_after - target_time):
            best_match_data = data[index - 1]
        else:
            best_match_data = data[index]
    return {
        # best_match_data[0] là 'n' (tên frame), [1] là 'frame_idx'
        "frame_n": best_match_data[0], 
        "frame_idx": best_match_data[1]
    }
# --- KẾT THÚC HÀM HELPER ---

# --- 8. CÁC API ---

# (XÓA BỎ) Hàm helper get_request_data()
# def get_request_data(): ...

# (CẬP NHẬT) Hàm chuẩn hóa đường dẫn web
def get_web_path(original_path):
    return parse_keyframe_path(original_path)


def get_frame_web_path(video_id, frame_id):
    meta = metadata_cache.get(video_id, {}).get(int(frame_id), {})
    return meta.get('path')


def get_neighbor_frame_ids(video_id, frame_id, radius):
    frames = video_frame_ids.get(video_id, [])
    try:
        position = frames.index(int(frame_id))
    except ValueError:
        return []
    start = max(0, position - radius)
    end = min(len(frames), position + radius + 1)
    return frames[start:end]


# === (THÊM MỚI) QUERY EXPANSION (Groq) - Theo "[AIC2026] - Query expansion.docx", PLAN A ===
QUERY_EXPANSION_PROMPT_TEMPLATE = """Bạn là chuyên gia viết truy vấn cho mô hình Jina đa phương thức trong bài toán Video Information Retrieval.

Nhiệm vụ: Chuyển câu truy vấn tiếng Việt thành 3 biến thể tiếng Việt giàu chi tiết thị giác để đối chiếu với cả ảnh và caption tiếng Anh bằng Jina.

QUY TẮC BẮT BỘC:
1. LOẠI BỎ TỪ KHÔNG CÓ HÌNH ẢNH: Bỏ các từ chỉ cảm xúc ("vui vẻ", "thanh mát"), từ chỉ nhiệm vụ ("làm nhiệm vụ", "nghiên cứu"), địa danh chung chung ("miền Nam", "miền Tây").
2. THỊ GIÁC HÓA (Visual Concretization): Dịch các khái niệm thành mô tả hình ảnh trực quan (Ví dụ: "loài chim ở Nam Bộ" -> dịch chi tiết đặc điểm màu lông, màu mắt của chim được mô tả trong câu).
3. BẢO TOÀN THỰC THỂ (Entity Recall): KHÔNG ĐƯỢC BỎ SÓT bất kỳ đối tượng, màu sắc, trang phục, đồ vật phụ nào (như "khăn rằn", "ghe xanh", "hoa pansy", "chiếc túi giấy", "hộp đổ bóng").
4. GIỮ NGUYÊN Ý NGHĨA: KHÔNG BỊẠ THÊM các chi tiết không có trong câu gốc.

Ví dụ mẫu:
Input: "Cảnh thu hoạch dứa: một bà cụ ngồi bên giỏ dứa trò chuyện với cô gái mặc áo hồng quàng khăn rằn; xung quanh chất đầy dứa, phía sau có người phụ nữ đội nón lá cầm trái dứa và một chiếc ghe xanh đậu cạnh bờ."

JSON Output:
{{
  "dense_caption": "An elderly woman sitting next to a basket of pineapples talking to a girl wearing a pink shirt and a traditional checked scarf, surrounded by harvested pineapples, with a woman in a conical hat holding a pineapple behind them and a blue boat parked by the riverbank.",
  "structured_entities": "elderly woman, basket of pineapples, girl in pink shirt, checked scarf, woman in conical hat, blue boat, riverbank, harvested pineapples",
  "spatial_action_focus": "a girl in pink shirt and an elderly woman sitting near pineapples with a blue boat moored at the shore"
}}

Yêu cầu đầu ra cho câu truy vấn dưới đây:
- "dense_caption": Dịch toàn bộ câu sang tiếng Anh Alt-text tự nhiên, giữ lại 100% chi tiết thị giác, màu sắc, vị trí.
- "structured_entities": Liệt kê TẤT CẢ các cụm thực thể + thuộc tính (màu sắc, hình dáng) ngăn cách bằng dấu phẩy.
- "spatial_action_focus": Tóm tắt mối quan hệ không gian và hành động cốt lõi giữa các chủ thể chính.

Câu truy vấn gốc: "{query}"

Chỉ trả về 1 Object JSON duy nhất, không thêm bất kỳ dòng giải thích hay ký tự markdown nào khác:
{{"dense_caption": "...", "structured_entities": "...", "spatial_action_focus": "..."}}"""

def expand_query_with_groq(query_text):
    """Mở rộng query bảo toàn tối đa chi tiết cho Jina image/caption retrieval."""
    if not groq_client:
        return []
    try:
        prompt = QUERY_EXPANSION_PROMPT_TEMPLATE.format(query=query_text)
        raw = groq_generate(prompt).strip()
        raw = re.sub(r'^```(json)?|```$', '', raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)

        # Lấy đầy đủ 3 biến thể giàu chi tiết
        variants = [
            parsed.get('dense_caption', ''),
            parsed.get('structured_entities', ''),
            parsed.get('spatial_action_focus', '')
        ]

        # Lọc bỏ chuỗi rỗng
        variants = [v.strip() for v in variants if v and v.strip()]

        # Loại bỏ các biến thể trùng lặp nếu có
        variants = list(dict.fromkeys(variants))

        print(f"[QueryExpansion] Raw: '{query_text}'")
        for idx, var in enumerate(variants, 1):
            print(f"  └─ Variant {idx}: {var}")

        return variants
    except Exception as e:
        print(f"Lỗi khi mở rộng câu truy vấn bằng Groq: {e}")
        return []

# (THÊM MỚI) API /expand_query - chỉ sinh 3 biến thể để người dùng chọn, KHÔNG tự search.
# Trước đây tick checkbox là tự động search cả 3 biến thể + gộp RRF (người dùng không biết đã tìm
# bằng câu gì). Giờ tách riêng: bấm nút "Mở rộng" -> hiện 3 lựa chọn -> người dùng bấm chọn 1 cái ->
# cái đó trở thành query rồi search bình thường qua /search.
@app.route('/expand_query', methods=['POST'])
def expand_query_endpoint():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request phải là JSON"}), 400
        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({"variants": []})
        variants = expand_query_with_groq(query_text)
        return jsonify({"variants": variants})
    except Exception as e:
        print(f"Lỗi trong /expand_query: {e}")
        return jsonify({"error": str(e)}), 500

def reciprocal_rank_fusion(ranked_id_lists, k=60, weights=None):
    """RRF: mỗi list là danh sách index đã sort tốt nhất trước. Trả (idx, fused_score) sort giảm dần.
    weights (tuỳ chọn): trọng số tương ứng từng list theo thứ tự trong ranked_id_lists, mặc định bằng nhau
    (dùng cho Fusion search: tỷ lệ Jina Hybrid/OCR/ASR do người dùng chỉnh)."""
    if weights is None:
        weights = [1.0] * len(ranked_id_lists)
    scores = {}
    for w, ranked_ids in zip(weights, ranked_id_lists):
        for rank, idx in enumerate(ranked_ids):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
# === (KẾT THÚC) QUERY EXPANSION ===

# === (CẬP NHẬT) OCR/ASR: bỏ hẳn Elasticsearch, dùng thẳng BM25 tự viết (đã build sẵn lúc khởi động) ===
def coverage_phrase_bm25_scores(bm25_index, corpus, tokenized_query, search_size):
    """BM25 scores with length-independent term coverage and exact phrase bonus."""
    tokenized_query = list(dict.fromkeys(tokenized_query))
    if not tokenized_query:
        return None

    scores, match_counts = bm25_index.get_scores_with_match_counts(tokenized_query)
    known_query_terms = [term for term in tokenized_query if term in bm25_index.idf]
    if not known_query_terms:
        return None

    query_weight = sum(bm25_index.idf[term] for term in known_query_terms)
    coverage = match_counts.astype(np.float64) / len(known_query_terms)
    scores += query_weight * np.square(coverage)

    # Chỉ dò vị trí phrase trong một pool rộng sau coverage để request vẫn nhanh.
    if len(tokenized_query) > 1:
        rerank_size = min(
            bm25_index.doc_count,
            max(int(search_size) * 20, 2000),
        )
        if rerank_size < bm25_index.doc_count:
            rerank_indices = np.argpartition(scores, -rerank_size)[-rerank_size:]
        else:
            rerank_indices = np.arange(bm25_index.doc_count)
        phrase_length = len(tokenized_query)
        phrase_bonus = query_weight * 1.5
        for index in rerank_indices:
            document = corpus[int(index)]
            if any(
                document[start:start + phrase_length] == tokenized_query
                for start in range(len(document) - phrase_length + 1)
            ):
                scores[int(index)] += phrase_bonus
    return scores


def ocr_candidates(query_text, search_size):
    """OCR ranking: BM25 nhẹ length penalty + query coverage + phrase bonus."""
    if not bm25_ocr_index:
        return []
    scores = coverage_phrase_bm25_scores(
        bm25_ocr_index,
        tokenized_corpus_ocr,
        tokenize_ocr_text(query_text),
        search_size,
    )
    if scores is None:
        return []

    top_k_indices = np.argsort(scores)[::-1][:search_size]
    out = []
    for i in top_k_indices:
        score = scores[i]
        if score <= 0:
            continue
        path = ocr_data[i].get('path', '')
        if not path:
            continue
        out.append((float(score), path))
    return out


def asr_candidates(query_text, search_size):
    """ASR ranking dùng cùng coverage/phrase policy với OCR."""
    if not bm25_asr_index:
        return []
    scores = coverage_phrase_bm25_scores(
        bm25_asr_index,
        asr_corpus_tokenized,
        tokenize_asr_text(query_text),
        search_size,
    )
    if scores is None:
        return []
    top_k_indices = np.argsort(scores)[::-1][:search_size]
    out = []
    for i in top_k_indices:
        score = scores[i]
        if score <= 0:
            continue
        doc = asr_data[i]
        out.append({
            "video_id": doc["video_id"],
            "text": doc["text"],
            "start": doc["start"],
            "end": doc["end"],
            "score": float(score),
        })
    return out
# === (KẾT THÚC) OCR/ASR SEARCH HELPERS ===


SEMANTIC_MODEL_LABELS = {
    "jina": "Jina Embeddings v5",
    "jina-hybrid": "Jina Image + Caption (RRF)",
}


def encode_semantic_query(query_text, semantic_model):
    if semantic_model in {"jina", "jina-hybrid"}:
        return jina_text_encoder.encode(query_text)
    raise ValueError(
        f"semantic_model không hợp lệ: {semantic_model!r}. "
        f"Chọn một trong {sorted(SEMANTIC_MODEL_LABELS)}."
    )


def search_semantic_vectors(semantic_model, query_vector, top_k):
    if semantic_model == "jina":
        return jina_semantic_index.search(query_vector, top_k)
    if semantic_model == "jina-hybrid":
        if jina_caption_index is None:
            raise ModelUnavailableError(jina_caption_index_reason)

        # Hai nhánh có phân phối cosine khác nhau (image vs caption), nên gộp
        # thứ hạng bằng RRF thay vì cộng trực tiếp raw similarity score.
        # Normal search benefits from a wider pool; TRAKE already asks for
        # 10k candidates, so cap here instead of expanding to 50k per branch.
        branch_k = max(min(int(top_k) * 5, 10000), 100)
        _, image_indices = jina_semantic_index.search(query_vector, branch_k)
        _, caption_indices = jina_caption_index.search(query_vector, branch_k)
        image_rank = [int(index_id) for index_id in image_indices[0] if index_id >= 0]
        caption_rank = [
            int(index_id) for index_id in caption_indices[0] if index_id >= 0
        ]
        fused = reciprocal_rank_fusion(
            [image_rank, caption_rank],
            weights=[1.0, 1.0],
        )[: max(1, int(top_k))]
        return (
            np.asarray([[score for _, score in fused]], dtype=np.float32),
            np.asarray([[index_id for index_id, _ in fused]], dtype=np.int64),
        )
    raise ValueError(f"semantic_model không hợp lệ: {semantic_model!r}")


@app.route('/semantic_models', methods=['GET'])
def semantic_models_status():
    jina_available, jina_reason = jina_text_encoder.availability()
    caption_available = jina_available and jina_caption_index is not None
    caption_reason = (
        jina_caption_index_reason if jina_available else jina_reason
    )
    return jsonify({
        "models": {
            "jina": {
                "label": SEMANTIC_MODEL_LABELS["jina"],
                "available": jina_available,
                "dimension": jina_semantic_index.d,
                "vectors": jina_semantic_index.ntotal,
                "reason": jina_reason,
            },
            "jina-hybrid": {
                "label": SEMANTIC_MODEL_LABELS["jina-hybrid"],
                "available": caption_available,
                "dimension": 1024,
                "vectors": (
                    jina_caption_index.ntotal if jina_caption_index is not None else 0
                ),
                "reason": caption_reason,
            },
        }
    })

# API /search
# (TRONG app.py)
# API /search (ĐÃ CẬP NHẬT)
@app.route('/search', methods=['POST'])
def search():
    try:
        # 1. Lấy dữ liệu request
        data = request.get_json()
        if data is None:
             return jsonify({"error": "Request phải là JSON"}), 400
             
        query_text = str(data.get('query', '')).strip()
        if not query_text:
            return jsonify({"error": "Query không được để trống."}), 400
        top_k = int(data.get('top_k', 50))
        semantic_model = str(data.get('semantic_model', 'jina')).strip().lower()
        if semantic_model not in SEMANTIC_MODEL_LABELS:
            return jsonify({
                "error": f"semantic_model không hợp lệ: {semantic_model!r}",
                "allowed_models": sorted(SEMANTIC_MODEL_LABELS),
            }), 400
        # (SỬA LỖI) Xử lý 'group' (là boolean true/false)
        group_results = data.get('group', False) 
        
        # === (LOGIC MỚI) KIỂM TRA XEM QUERY CÓ PHẢI LÀ VIDEO ID KHÔNG ===
        
        # Chuẩn hóa query (ví dụ: " l22_v002 " -> "L22_V002")
        video_id_query = query_text.strip().upper() 
        
        # Kiểm tra xem query này có nằm trong danh sách video ID ta có không
        # Video IDs are discovered directly from OCR metadata.
        if video_id_query in metadata_cache:
            print(f"Video ID search detected: {video_id_query}")
            results = []
            summary = {}
            video_meta = metadata_cache.get(video_id_query, {})

            for frame_id, meta in video_meta.items():
                web_path = meta.get('path')
                if not web_path:
                    continue
                pts_time = float(meta.get('pts_time', 0) or 0)
                results.append({
                    "path": web_path,
                    "videoId": video_id_query,
                    "score": pts_time,
                    "pts_time": pts_time
                })

            summary[video_id_query] = len(results)
            final_results = sorted(results, key=lambda item: item['pts_time'])[:top_k]
            if group_results:
                return jsonify({
                    "results": {video_id_query: final_results},
                    "summary": summary
                })
            return jsonify({"results": final_results, "summary": summary})

        # === (KẾT THÚC LOGIC MỚI) ===
        
        # Nếu không phải là Video ID, chạy logic tìm kiếm semantic CŨ
        print(
            f"Đang tìm kiếm semantic bằng {SEMANTIC_MODEL_LABELS[semantic_model]} "
            f"cho: '{query_text}'"
        )

        # Jina nhận query tiếng Việt trực tiếp; caption tiếng Anh vẫn nằm trong
        # cùng không gian multilingual nên không cần dịch trước khi encode.
        search_query = query_text

        query_vector = encode_semantic_query(search_query, semantic_model)

        pool_k = top_k * 5 if group_results else top_k

        semantic_score_by_idx = {}
        distances, indices = search_semantic_vectors(semantic_model, query_vector, pool_k)
        ordered_indices = [int(i) for i in indices[0] if int(i) >= 0]
        for i, dist in zip(indices[0], distances[0]):
            if int(i) >= 0:
                semantic_score_by_idx[int(i)] = float(dist)

        results = []
        summary = {}

        for i in ordered_indices:
            original_path = image_records[int(i)]['path']
            web_path, video_id, frame_n_str = get_web_path(original_path)

            # (SỬA LỖI) Thêm check frame_n_str (không phải None)
            if web_path and frame_n_str:
                frame_n_int = int(frame_n_str)
                meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0

                results.append({
                    "path": web_path,
                    "videoId": video_id,
                    "score": semantic_score_by_idx.get(i, 0.0),
                    "pts_time": float(pts_time)
                })
                summary[video_id] = summary.get(video_id, 0) + 1

        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in results:
                video_id = res['videoId']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)
            
            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                sorted_items = sorted(items, key=lambda x: x['pts_time'])
                final_grouped_results[video_id] = sorted_items[:top_k] 
            
            return jsonify({
                "results": final_grouped_results,
                "summary": sorted_summary,
                "semantic_model": semantic_model,
            })
        else:
            final_results = results[:top_k]
            return jsonify({
                "results": final_results,
                "summary": sorted_summary,
                "semantic_model": semantic_model,
            })

    except ModelUnavailableError as e:
        print(f"Model semantic chưa sẵn sàng: {e}")
        return jsonify({"error": str(e), "semantic_model": semantic_model}), 503
    except ValueError as e:
        print(f"Request /search không hợp lệ: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Lỗi trong /search: {e}")
        return jsonify({"error": str(e)}), 500

# (CẬP NHẬT) API /search_similar_image - Tích hợp YOLOv8n Auto-Crop
@app.route('/search_similar_image', methods=['POST'])
def search_similar_image():
    try:
        # Lấy FormData trực tiếp
        data = request.form.to_dict()
        top_k = int(data.get('top_k', 50))
        group_results = data.get('group', 'false').lower() == 'true'
        # (THÊM MỚI) Tham số auto_crop từ nút Toggle trên giao diện
        auto_crop = data.get('auto_crop', 'false').lower() == 'true'
        if 'image_file' not in request.files:
            return jsonify({"error": "Không có tệp ảnh nào được tải lên."}), 400

        file = request.files['image_file']
        original_image = Image.open(io.BytesIO(file.read())).convert("RGB")
        target_image = original_image  # Mặc định dùng ảnh gốc

        # (THÊM MỚI) --- BƯỚC TIỀN XỬ LÝ: YOLO AUTO-CROP ---
        if auto_crop and yolo_model is not None:
            print("[AutoCrop] Đang detect vật thể bằng YOLOv8n...")
            with torch.no_grad():
                yolo_results = yolo_model(original_image, verbose=False)

            boxes = yolo_results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # --- Chiến lược chọn box: ưu tiên diện tích lớn gần trung tâm ---
                img_w, img_h = original_image.size
                img_cx, img_cy = img_w / 2.0, img_h / 2.0

                best_box = None
                best_score = -1.0

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    area = (x2 - x1) * (y2 - y1)
                    box_cx = (x1 + x2) / 2.0
                    box_cy = (y1 + y2) / 2.0

                    # Khoảng cách từ trung tâm box đến trung tâm ảnh (chuẩn hóa)
                    dist_to_center = ((box_cx - img_cx) ** 2 + (box_cy - img_cy) ** 2) ** 0.5
                    max_dist = ((img_w / 2) ** 2 + (img_h / 2) ** 2) ** 0.5
                    center_score = 1.0 - (dist_to_center / max_dist) if max_dist > 0 else 1.0

                    # Diện tích chuẩn hóa
                    area_normalized = area / (img_w * img_h) if (img_w * img_h) > 0 else 0

                    # Tổng hợp: 50% confidence + 30% diện tích + 20% gần trung tâm
                    combined_score = 0.5 * conf + 0.3 * area_normalized + 0.2 * center_score

                    if combined_score > best_score:
                        best_score = combined_score
                        best_box = box

                if best_box is not None:
                    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                    cls_id = int(best_box.cls[0])
                    conf = float(best_box.conf[0])
                    class_name = yolo_results[0].names.get(cls_id, f"class_{cls_id}")

                    # Padding 15px, clamp trong giới hạn ảnh
                    PADDING = 15
                    x1_pad = max(0, int(x1) - PADDING)
                    y1_pad = max(0, int(y1) - PADDING)
                    x2_pad = min(img_w, int(x2) + PADDING)
                    y2_pad = min(img_h, int(y2) + PADDING)

                    target_image = original_image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
                    print(f"[AutoCrop] Đã crop vật thể '{class_name}' với độ tin cậy {conf:.2f} | Box: [{x1_pad},{y1_pad},{x2_pad},{y2_pad}]")
                else:
                    print("[AutoCrop] Không tìm được box tốt nhất. Dùng ảnh gốc.")
            else:
                print("[AutoCrop] YOLO không phát hiện vật thể nào. Fallback về ảnh gốc.")

            # Giải phóng VRAM sau YOLO inference
            del yolo_results
            if device == "cuda":
                torch.cuda.empty_cache()
        # --- KẾT THÚC BƯỚC TIỀN XỬ LÝ ---

        # Ảnh query và toàn bộ keyframe đều dùng cùng Jina retrieval space.
        query_vector = jina_text_encoder.encode_image(target_image)

        # Dọn dẹp sau Jina inference
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        pool_k = top_k * 5 if group_results else top_k
        distances, indices = jina_semantic_index.search(query_vector, pool_k)
        ordered = [(int(i), float(dist)) for i, dist in zip(indices[0], distances[0])]

        results = []
        summary = {}

        for i, score in ordered:
            original_path = image_records[int(i)]['path']
            web_path, video_id, frame_n_str = get_web_path(original_path)

            if web_path and frame_n_str:
                frame_n_int = int(frame_n_str)
                meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0

                results.append({
                    "path": web_path,
                    "videoId": video_id,
                    "score": float(score),
                    "pts_time": float(pts_time)
                })
                summary[video_id] = summary.get(video_id, 0) + 1

        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in results:
                video_id = res['videoId']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)

            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                sorted_items = sorted(items, key=lambda x: x['pts_time'])
                final_grouped_results[video_id] = sorted_items[:top_k]

            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            final_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
            return jsonify({"results": final_results, "summary": sorted_summary})

    except ModelUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"Lỗi trong /search_similar_image: {e}")
        return jsonify({"error": str(e)}), 500


# (CẬP NHẬT) API /search_ocr
@app.route('/search_ocr', methods=['POST'])
def search_ocr():
    try:
        # (SỬA LỖI) Lấy JSON trực tiếp
        data = request.get_json()
        if data is None:
             return jsonify({"error": "Request phải là JSON"}), 400
             
        query_text = data['query'].lower()
        top_k = int(data.get('top_k', 50))
        # (SỬA LỖI) Xử lý 'group' (là boolean true/false)
        group_results = data.get('group', False)

        if not query_text:
            return jsonify({"results": [], "summary": {}})

        search_size = top_k * 5 if group_results else top_k

        candidates = ocr_candidates(query_text, search_size)
        if not candidates and not bm25_ocr_index:
            return jsonify({"error": "Chưa có dữ liệu OCR (bm25_ocr_index chưa khởi tạo)."}), 500

        results = []
        summary = {}

        for score, original_path in candidates:
            # Chạy tiếp luồng xử lý Web Path và Metadata của ông
            web_path, video_id, frame_n_str = get_web_path(original_path)

            if web_path and frame_n_str:
                frame_n_int = int(frame_n_str)
                meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0

                results.append({
                    "path": web_path,
                    "videoId": video_id,
                    "score": float(score),
                    "pts_time": float(pts_time)
                })
                summary[video_id] = summary.get(video_id, 0) + 1
        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in results:
                video_id = res['videoId']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)
            
            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                sorted_items = sorted(items, key=lambda x: x['pts_time'])
                final_grouped_results[video_id] = sorted_items[:top_k]
            
            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            final_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
            return jsonify({"results": final_results, "summary": sorted_summary})
            
    except Exception as e: 
        print(f"Lỗi trong /search_ocr: {e}")
        return jsonify({"error": str(e)}), 500

# === (VIẾT LẠI) TRAKE - TEMPORAL ALIGNMENT ===
# Bản cũ tìm frame mà TẤT CẢ các phần mô tả cùng xuất hiện trong một cửa sổ hẹp (±window_size frame).
# Điều đó sai bản chất TRAKE: các sự kiện trong một chuỗi (chạy đà -> giậm nhảy -> bay qua xà ->
# tiếp đất) nằm RẢI RÁC theo thời gian, có thể cách nhau hàng trăm frame, nên gần như không bao giờ
# "giao nhau" trong một cửa sổ hẹp.
#
# Bản mới làm đúng 2 giai đoạn theo mô tả của BTC:
#   1. Retrieval  - tìm video chứa toàn bộ chuỗi sự kiện (gộp điểm mọi sự kiện theo từng video).
#   2. Alignment  - trong mỗi video, chọn cho mỗi sự kiện đúng 1 keyframe sao cho thứ tự thời gian
#                   được giữ nguyên và tổng điểm Jina Hybrid lớn nhất,
#                   ĐỒNG THỜI phạt nặng khi 2 sự kiện liên tiếp cách nhau quá xa về thời gian.
# Giai đoạn 2 là bài toán quy hoạch động (DP) - xem _align_event_sequence().
TRAKE_DEFAULT_MAX_GAP_SECONDS = 30.0   # trong khoảng này thì không phạt
TRAKE_DEFAULT_GAP_PENALTY = 0.01       # phạt mỗi giây vượt ngưỡng; vượt 30s ~ mất trọn 1 sự kiện khớp


def _align_event_sequence(candidate_frames, frame_times, event_scores, n_events,
                          max_gap_seconds=TRAKE_DEFAULT_MAX_GAP_SECONDS,
                          gap_penalty_per_sec=TRAKE_DEFAULT_GAP_PENALTY):
    """Chọn chuỗi frame tăng dần, tối đa hoá tổng điểm Jina Hybrid trừ tiền phạt
    khoảng cách thời gian giữa 2 sự kiện liên tiếp.

    Một chuỗi hành động (chạy đà -> giậm nhảy -> ...) diễn ra liên tục trong vài chục giây, nên nếu
    chỉ ràng buộc "frame sau > frame trước" thì DP hay ghép các sự kiện cách nhau vài phút - vốn là
    những cảnh không liên quan trong cùng video. Tiền phạt tuyến tính phần vượt quá max_gap_seconds
    khiến các chuỗi rời rạc như vậy tụt hạng, nhưng vẫn không loại hẳn (phòng khi đáp án thật hơi thưa).

    candidate_frames: list frame_n (int) đã sort tăng dần - các keyframe ứng viên của 1 video.
    frame_times:      list pts_time (giây) song song với candidate_frames, cũng tăng dần.
    event_scores:     dict[(event_index, frame_n)] -> điểm Jina Hybrid. Thiếu key = sự kiện đó không
                      khớp frame đó (tính 0 điểm, vẫn cho chọn để giữ chuỗi liền mạch).
    Trả (total_score, [frame_n cho từng sự kiện]) hoặc None nếu không xếp được chuỗi hợp lệ.
    """
    m = len(candidate_frames)
    if m < n_events:
        return None  # không đủ frame để xếp N mốc khác nhau theo thứ tự tăng dần

    NEG = -1e18  # dùng số hữu hạn thay -inf để tránh NaN khi trừ tiền phạt
    times = np.asarray(frame_times, dtype=np.float64)

    # pen[i][k] = tiền phạt khi nhảy từ frame k sang frame i (chỉ tính phần vượt quá max_gap_seconds)
    gaps = times[:, None] - times[None, :]
    pen = np.maximum(gaps - max_gap_seconds, 0.0) * gap_penalty_per_sec
    # Chỉ cho phép k < i để giữ đúng thứ tự thời gian của chuỗi sự kiện
    allowed = np.tril(np.ones((m, m), dtype=bool), k=-1)

    dp = np.full((n_events, m), NEG, dtype=np.float64)
    parent = np.full((n_events, m), -1, dtype=np.int64)

    dp[0] = np.array([event_scores.get((0, fn), 0.0) for fn in candidate_frames], dtype=np.float64)

    for j in range(1, n_events):
        # vals[i][k] = điểm tốt nhất tới sự kiện j-1 tại frame k, trừ tiền phạt khi nhảy sang frame i
        vals = np.where(allowed, dp[j - 1][None, :] - pen, NEG)
        best_prev_i = vals.argmax(axis=1)
        best_prev = vals[np.arange(m), best_prev_i]
        own = np.array([event_scores.get((j, fn), 0.0) for fn in candidate_frames], dtype=np.float64)
        no_path = best_prev <= NEG / 2  # không có frame hợp lệ nào đứng trước
        dp[j] = np.where(no_path, NEG, best_prev + own)
        parent[j] = np.where(no_path, -1, best_prev_i)

    last = n_events - 1
    best_i = int(dp[last].argmax())
    if dp[last][best_i] <= NEG / 2:
        return None

    chosen = [None] * n_events
    i = best_i
    for j in range(last, -1, -1):
        chosen[j] = candidate_frames[i]
        i = int(parent[j][i])
        if i < 0 and j > 0:
            return None  # chuỗi truy vết bị đứt (không nên xảy ra)
    return float(dp[last][best_i]), chosen


@app.route('/search_trake_02', methods=['POST'])
def search_trake_02():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request phải là JSON"}), 400

        top_k_final = int(data.get('top_k', 50))
        # Mỗi phần tử là MỘT SỰ KIỆN, theo đúng thứ tự thời gian trong video.
        # UI gửi mảng 'events' (mỗi sự kiện một ô nhập riêng); vẫn chấp nhận chuỗi 'query'
        # ngăn bằng ';' để gọi API trực tiếp bằng script/curl cho tiện.
        raw_events = data.get('events')
        if isinstance(raw_events, list):
            parts = [str(p).strip() for p in raw_events if str(p).strip()]
        else:
            parts = [p.strip() for p in data.get('query', '').split(';') if p.strip()]

        if len(parts) < 2:
            return jsonify({"error": "TRAKE cần ít nhất 2 sự kiện "
                                     "(ví dụ: 'chạy đà', 'giậm nhảy', 'bay qua xà', 'tiếp đất')"}), 400

        n_events = len(parts)
        # Số sự kiện tối thiểu phải thực sự khớp thì video mới được giữ lại (cho phép hụt 1 sự kiện)
        min_events = int(data.get('min_events', max(2, n_events - 1)))
        # Ngưỡng khoảng cách thời gian giữa 2 sự kiện liên tiếp: trong ngưỡng thì không phạt,
        # vượt bao nhiêu giây thì trừ điểm bấy nhiêu * gap_penalty_per_sec.
        max_gap_seconds = float(data.get('max_gap_seconds', TRAKE_DEFAULT_MAX_GAP_SECONDS))
        gap_penalty_per_sec = float(data.get('gap_penalty_per_sec', TRAKE_DEFAULT_GAP_PENALTY))
        pool_k = max(top_k_final * 100, 10000)

        print(f"[TRAKE] Temporal alignment: {n_events} sự kiện, pool_k={pool_k}, "
              f"min_events={min_events}, max_gap={max_gap_seconds}s, penalty={gap_penalty_per_sec}/s")

        # video_event_scores[video_id][(event_index, frame_n)] = điểm Hybrid tốt nhất
        video_event_scores = collections.defaultdict(dict)
        # video_matched_events[video_id] = tập các event_index thực sự có hit trong video đó
        video_matched_events = collections.defaultdict(set)

        # Jina đa ngôn ngữ nhận trực tiếp toàn bộ sự kiện tiếng Việt.
        event_queries = parts
        for i, q in enumerate(event_queries):
            print(f"  [Sự kiện {i + 1}/{n_events}] '{q}'")

        query_vectors = jina_text_encoder.encode_texts(event_queries)
        for event_index in range(n_events):
            distances, indices = search_semantic_vectors(
                "jina-hybrid", query_vectors[event_index:event_index + 1], pool_k
            )
            for idx, dist in zip(indices[0], distances[0]):
                idx = int(idx)
                if idx < 0:
                    continue
                web_path, video_id, frame_n_str = get_web_path(image_records[int(idx)]['path'])
                if not web_path or not frame_n_str or video_id == "N/A":
                    continue
                frame_n = int(frame_n_str)
                key = (event_index, frame_n)
                # RRF tối đa xấp xỉ 2/61. Scale về gần [0, 1] để giữ nguyên
                # ý nghĩa của gap_penalty_per_sec trong thuật toán alignment.
                score = float(dist) * 30.5
                scores_of_video = video_event_scores[video_id]
                if score > scores_of_video.get(key, float('-inf')):
                    scores_of_video[key] = score
                video_matched_events[video_id].add(event_index)

        # Giai đoạn 2: căn chỉnh thời gian trong từng video ứng viên
        sequences = []
        for video_id, event_scores in video_event_scores.items():
            if len(video_matched_events[video_id]) < min_events:
                continue

            video_meta = metadata_cache.get(video_id, {})
            # Chỉ giữ frame có pts_time thật - không có mốc thời gian thì không tính được khoảng cách
            candidate_frames, candidate_times = [], []
            for frame_n in sorted({frame_n for _, frame_n in event_scores.keys()}):
                meta = video_meta.get(frame_n) or {}
                pts_time = meta.get('pts_time')
                if pts_time is None:
                    continue
                candidate_frames.append(frame_n)
                candidate_times.append(float(pts_time))
            if len(candidate_frames) < n_events:
                continue

            aligned = _align_event_sequence(candidate_frames, candidate_times, event_scores, n_events,
                                            max_gap_seconds=max_gap_seconds,
                                            gap_penalty_per_sec=gap_penalty_per_sec)
            if aligned is None:
                continue
            total_score, chosen_frames = aligned

            events_out = []
            matched_count = 0
            prev_time = None
            for event_index, frame_n in enumerate(chosen_frames):
                is_matched = (event_index, frame_n) in event_scores
                if is_matched:
                    matched_count += 1
                meta = video_meta.get(frame_n, {}) or {}
                pts_time = float(meta.get('pts_time', 0) or 0)
                frame_idx = meta.get('frame_idx')
                # Khoảng cách tới sự kiện liền trước -> UI cảnh báo khi vượt ngưỡng
                gap_from_prev = None if prev_time is None else round(pts_time - prev_time, 2)
                prev_time = pts_time
                events_out.append({
                    "event_index": event_index,
                    "query": parts[event_index],
                    "path": get_frame_web_path(video_id, frame_n),
                    "videoId": video_id,
                    "frame_n": int(frame_n),
                    # frame_idx = chỉ số frame thật trong video -> dùng thẳng để nộp đáp án TRAKE
                    "frame_idx": int(frame_idx) if frame_idx is not None else None,
                    "pts_time": pts_time,
                    "gap_from_prev": gap_from_prev,
                    "score": float(event_scores.get((event_index, frame_n), 0.0)),
                    "matched": is_matched
                })

            sequences.append({
                "videoId": video_id,
                "score": float(total_score),
                "matched_events": matched_count,
                "total_events": n_events,
                # Tổng thời lượng chuỗi - chuỗi càng gọn càng đáng tin cho một hành động liên tục
                "span_seconds": round(events_out[-1]["pts_time"] - events_out[0]["pts_time"], 2),
                "events": events_out
            })

        sequences.sort(key=lambda s: -s['score'])
        final_sequences = sequences[:top_k_final]
        print(f"[TRAKE] {len(sequences)} video xếp được chuỗi hợp lệ, trả về {len(final_sequences)}")

        summary = {s['videoId']: s['matched_events'] for s in final_sequences}
        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        return jsonify({
            "results": final_sequences,
            "summary": sorted_summary,
            "events_query": parts,
            "max_gap_seconds": max_gap_seconds,
            "mode": "trake_temporal"
        })

    except ModelUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"Lỗi trong /search_trake_02: {e}")
        return jsonify({"error": str(e)}), 500
# (THÊM MỚI) API /search_trake_image (TRAKE.02 với nhiều ảnh)
@app.route('/search_trake_image', methods=['POST'])
def search_trake_image():
    try:
        # Lấy FormData trực tiếp (hỗ trợ multiple files)
        data = request.form.to_dict()
        top_k_final = int(data.get('top_k', 50))
        group_results = data.get('group', 'false').lower() == 'true'
        
        # Lấy multiple images
        image_files = request.files.getlist('image_files')  # Array of files
        if not image_files or len(image_files) < 2:
            return jsonify({"error": "Cần ít nhất 2 ảnh để tìm giao (TRAKE.02 Image)."}), 400
        
        # (CẬP NHẬT) Ngưỡng "common points" (cho phép thiếu 1 ảnh nếu có nhiều ảnh)
        min_common = max(2, len(image_files) - 1) if len(image_files) >= 3 else len(image_files)
        
        # (CẬP NHẬT) Nới khung hình (Frame Windowing)
        window_size = int(data.get('window_size', 5)) # Mặc định nới +/- 5 frames
        
        # n1 (số frame cần lấy cho mỗi ảnh)
        top_k_per_image = max(top_k_final * 100, 10000)
        
        print(f"[TRAKE.02 Image] Đang tìm kiếm {len(image_files)} ảnh, n1={top_k_per_image}, min_common={min_common}, window_size=±{window_size}")
        
        # Store frame occurrences
        # Key: (video_id, frame_n_str), Value: Dict[img_index -> max_score]
        frame_image_scores = collections.defaultdict(lambda: collections.defaultdict(float))
        # Key: (video_id, frame_n_str), Value: {path, videoId, pts_time}
        frame_info_cache = {}
        # Key: video_id, Value: count (for summary)
        summary_counter = collections.defaultdict(int)
        
        for img_index, file in enumerate(image_files):
                # Xử lý từng ảnh
                if file.filename == '': continue
                image = Image.open(io.BytesIO(file.read())).convert("RGB")

                query_vector = jina_text_encoder.encode_image(image)
                
                # Tìm n1 frames cho ảnh này
                distances, indices = jina_semantic_index.search(
                    query_vector, top_k_per_image
                )
                
                for i, dist in zip(indices[0], distances[0]):
                    original_path = image_records[int(i)]['path']
                    web_path, video_id, frame_n_str = get_web_path(original_path)
                    
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        video_meta = metadata_cache.get(video_id, {})
                        
                        # (CẬP NHẬT) Quét cửa sổ +/- window_size
                        for neighbor_n in get_neighbor_frame_ids(video_id, frame_n_int, window_size):
                            frame_key = (video_id, neighbor_n)

                            current_best = frame_image_scores[frame_key].get(img_index, -1000.0)
                            if float(dist) > current_best:
                                frame_image_scores[frame_key][img_index] = float(dist)

                            if frame_key not in frame_info_cache:
                                meta = video_meta[neighbor_n]
                                neighbor_web_path = meta.get('path')
                                if not neighbor_web_path:
                                    continue
                                pts_time = float(meta.get('pts_time', 0) or 0)
                                frame_info_cache[frame_key] = {
                                    "path": neighbor_web_path,
                                    "videoId": video_id,
                                    "pts_time": pts_time
                                }

        # Lọc kết quả dựa trên số "common images"
        all_results = []
        for frame_key, imgs_dict in frame_image_scores.items():
            common_count = len(imgs_dict)
            if common_count >= min_common:
                # Frame này match với ít nhất `min_common` ảnh
                info = frame_info_cache[frame_key].copy()
                info['sum_score'] = sum(imgs_dict.values())
                info['score'] = info['sum_score']
                info['common_count'] = common_count
                all_results.append(info)
                summary_counter[info['videoId']] += 1
        
        print(f"[TRAKE.02 Image] Tìm thấy {len(all_results)} frame chung (>= {min_common} ảnh)")
        
        # Sắp xếp và định dạng output (giống TRAKE.02 text)
        sorted_summary = dict(sorted(summary_counter.items(), key=lambda item: item[1], reverse=True))
        
        if group_results:
            grouped_results = {}
            for res in all_results:
                video_id = res['videoId']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)
            
            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                # Sắp xếp theo common_count giảm dần, sau đó pts_time
                sorted_items = sorted(items, key=lambda x: (-x['score'], x['pts_time']))
                final_grouped_results[video_id] = sorted_items[:top_k_final]
            
            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            # Sắp xếp theo common_count giảm dần, sau đó pts_time
            final_results = sorted(all_results, key=lambda x: (-x['score'], x['pts_time']))
            return jsonify({"results": final_results[:top_k_final], "summary": sorted_summary})
            
    except ModelUnavailableError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        print(f"Lỗi trong /search_trake_image: {e}")
        return jsonify({"error": str(e)}), 500
# API /search_asr - tìm trực tiếp trên từng ASR segment bằng BM25.
@app.route('/search_asr', methods=['POST'])
def search_asr():
    try:
        data = request.get_json()
        if data is None:
             return jsonify({"error": "Request phải là JSON"}), 400
             
        query_text = data['query']
        top_k = int(data.get('top_k', 50))
        group_results = data.get('group', False)
        
        if not query_text:
            return jsonify({"results": [], "summary": {}})

        search_size = top_k * 5 if group_results else top_k * 2
        top_k_documents = asr_candidates(query_text, search_size)
        if not top_k_documents and not bm25_asr_index:
            return jsonify({"error": "Chưa có dữ liệu ASR (bm25_asr_index chưa khởi tạo)."}), 500

        # ASR hiện tại đã chia thành các segment dài, vì vậy trả từng segment
        # độc lập. Cơ chế stitch previous/current/next dành cho bộ ASR segment
        # ngắn trước đây đã được đưa vào backlog.
        final_results = []
        summary = {}

        for doc in top_k_documents:
            video_id = doc['video_id']
            final_doc = doc.copy()

            final_doc['watch_url'] = video_url_cache.get(video_id)

            target_start_time = final_doc['start']
            closest_frame_data = find_closest_keyframe(video_id, target_start_time)
            final_doc['frame_n']   = closest_frame_data.get('frame_n')
            final_doc['frame_idx'] = closest_frame_data.get('frame_idx')

            frame_n = final_doc['frame_n']
            final_doc['web_path'] = (
                get_frame_web_path(video_id, frame_n) if frame_n is not None else None
            )

            final_results.append(final_doc)
            summary[video_id] = summary.get(video_id, 0) + 1
        
        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in final_results:
                video_id = res['video_id']
                if video_id == "N/A": continue
                if video_id not in grouped_results:
                    grouped_results[video_id] = []
                grouped_results[video_id].append(res)
            
            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                # Sắp xếp ASR theo thời gian bắt đầu
                sorted_items = sorted(items, key=lambda x: x['start'])
                final_grouped_results[video_id] = sorted_items[:top_k]
            
            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            final_results = sorted(final_results, key=lambda x: x['score'], reverse=True)[:top_k]
            return jsonify({"results": final_results, "summary": sorted_summary})

    except Exception as e:
        print(f"Lỗi trong /search_asr: {e}")
        return jsonify({"error": str(e)}), 500


# API /search_fusion - Gộp Jina Hybrid + OCR + ASR bằng RRF có trọng số.
# Mỗi khoảnh khắc được định danh bằng (video_id, frame_n) để gộp điểm giữa 3 nguồn có thang đo khác nhau
# (Jina Hybrid: RRF image/caption, OCR/ASR: điểm BM25 - không thể cộng trực tiếp).
@app.route('/search_fusion', methods=['POST'])
def search_fusion():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request phải là JSON"}), 400

        query_jina = data.get('query_jina', '').strip()
        query_ocr = data.get('query_ocr', '').strip()
        query_asr = data.get('query_asr', '').strip()
        if not query_jina and not query_ocr and not query_asr:
            return jsonify({"results": [], "summary": {}})

        # Trọng số thô (mặc định bằng nhau) - chỉ tỷ lệ tương đối giữa 3 số này mới ảnh hưởng thứ hạng RRF,
        # nên không cần chuẩn hoá về tổng=1 trước khi tính.
        weight_jina = float(data.get('weight_jina', 1.0))
        weight_ocr = float(data.get('weight_ocr', 1.0))
        weight_asr = float(data.get('weight_asr', 1.0))

        top_k = int(data.get('top_k', 50))
        group_results = data.get('group', False)
        pool_k = max(top_k * 5, 100)

        jina_ranked_keys, ocr_ranked_keys, asr_ranked_keys = [], [], []
        key_info = {}
        key_sources = collections.defaultdict(set)

        def register(key, web_path, video_id, pts_time, source):
            key_sources[key].add(source)
            if key not in key_info:
                key_info[key] = {
                    "path": web_path,
                    "videoId": video_id,
                    "pts_time": float(pts_time) if pts_time else 0.0
                }

        # --- 1. Nhánh Jina Hybrid (Jina image + Jina caption) ---
        if query_jina:
            try:
                query_vector = encode_semantic_query(query_jina, "jina-hybrid")
                _, indices = search_semantic_vectors("jina-hybrid", query_vector, pool_k)
                for i in indices[0]:
                    i = int(i)
                    if i < 0:
                        continue
                    web_path, video_id, frame_n_str = get_web_path(image_records[int(i)]['path'])
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        key = (video_id, frame_n_int)
                        jina_ranked_keys.append(key)
                        meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                        register(key, web_path, video_id, meta.get('pts_time', 0) if meta else 0, "JINA_HYBRID")
            except Exception as e:
                print(f"[Fusion] Lỗi nhánh Jina Hybrid: {e}")

        # --- 2. Nhánh OCR (BM25 tự implement) ---
        if query_ocr:
            try:
                ocr_hits = ocr_candidates(query_ocr.lower(), pool_k)
                for score, original_path in ocr_hits:
                    web_path, video_id, frame_n_str = get_web_path(original_path)
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        key = (video_id, frame_n_int)
                        ocr_ranked_keys.append(key)
                        meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                        register(key, web_path, video_id, meta.get('pts_time', 0) if meta else 0, "OCR")
            except Exception as e:
                print(f"[Fusion] Lỗi nhánh OCR: {e}")

        # --- 3. Nhánh ASR (BM25 tự implement) ---
        if query_asr:
            try:
                asr_hits = asr_candidates(query_asr, pool_k)
                for src in asr_hits:
                    video_id = src.get("video_id", "")
                    if not video_id:
                        continue
                    closest = find_closest_keyframe(video_id, src.get("start", 0.0))
                    frame_n = closest.get('frame_n')
                    if frame_n is None:
                        continue
                    frame_n_int = int(frame_n)
                    web_path = get_frame_web_path(video_id, frame_n_int)
                    if not web_path:
                        continue
                    key = (video_id, frame_n_int)
                    asr_ranked_keys.append(key)
                    meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                    pts_time = meta.get('pts_time', 0) if meta else src.get("start", 0.0)
                    register(key, web_path, video_id, pts_time, "ASR")
            except Exception as e:
                print(f"[Fusion] Lỗi nhánh ASR: {e}")

        # --- 4. Gộp bằng RRF có trọng số ---
        branches = [
            (jina_ranked_keys, weight_jina),
            (ocr_ranked_keys, weight_ocr),
            (asr_ranked_keys, weight_asr),
        ]
        branches = [(lst, w) for lst, w in branches if lst]
        if not branches:
            return jsonify({"results": [], "summary": {}, "error": "Không có nhánh nào (Jina Hybrid/OCR/ASR) trả về kết quả."})

        ranked_lists = [lst for lst, _ in branches]
        weights = [w for _, w in branches]
        fused = reciprocal_rank_fusion(ranked_lists, weights=weights)

        results = []
        summary = {}
        for key, score in fused:
            info = key_info.get(key)
            if not info:
                continue
            video_id = info["videoId"]
            results.append({
                "path": info["path"],
                "videoId": video_id,
                "score": float(score),
                "pts_time": info["pts_time"],
                "matched_by": sorted(key_sources[key])
            })
            summary[video_id] = summary.get(video_id, 0) + 1

        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        if group_results:
            grouped_results = {}
            for res in results:
                video_id = res['videoId']
                if video_id == "N/A":
                    continue
                grouped_results.setdefault(video_id, []).append(res)

            final_grouped_results = {}
            for video_id, items in grouped_results.items():
                sorted_items = sorted(items, key=lambda x: x['pts_time'])
                final_grouped_results[video_id] = sorted_items[:top_k]

            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            return jsonify({"results": results[:top_k], "summary": sorted_summary})

    except Exception as e:
        print(f"Lỗi trong /search_fusion: {e}")
        return jsonify({"error": str(e)}), 500


# (CẬP NHẬT) API /metadata
@app.route('/metadata', methods=['POST'])
def get_metadata():
    try:
        image_path = request.json['image_path']
        _, video_id, frame_id_str = get_web_path(image_path)
        if not frame_id_str or video_id == "N/A":
            raise ValueError(f"Invalid keyframe path: {image_path}")
        frame_id = int(frame_id_str)
        meta = dict(metadata_cache.get(video_id, {}).get(frame_id, {}))
        meta['n'] = frame_id
        watch_url = video_url_cache.get(video_id)
        if watch_url and meta.get('pts_time') is not None:
            separator = '&' if '?' in watch_url else '?'
            meta['playback_url'] = f"{watch_url}{separator}t={int(float(meta['pts_time']))}s"
        else:
            meta['playback_url'] = watch_url
        return jsonify(meta)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# (CẬP NHẬT) API /neighbor_frames
@app.route('/neighbor_frames', methods=['POST'])
def get_neighbor_frames():
    try:
        image_path = request.json['image_path']
        _, video_id, frame_id_str = get_web_path(image_path)
        if not frame_id_str or video_id == "N/A":
            return jsonify({"neighbors": []})
        neighbor_ids = get_neighbor_frame_ids(video_id, int(frame_id_str), 5)
        neighbors = [get_frame_web_path(video_id, frame_id) for frame_id in neighbor_ids]
        return jsonify({"neighbors": [path for path in neighbors if path]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === (CẬP NHẬT) API ĐỂ LẤY BẢN ĐỒ THỜI GIAN KEYFRAME ===
@app.route('/get_keyframe_map', methods=['POST'])
def get_keyframe_map():
    try:
        video_id = request.json['video_id']
        map_data = keyframe_time_cache.get(video_id)
        # (SỬA LỖI) Thêm check `if map_data`
        if map_data:
            return jsonify(map_data)
        else:
            return jsonify({"error": f"Map data not found for video_id: {video_id}"}), 404
    # (SỬA LỖI) Thụt lề khối 'except'
    except Exception as e: 
        print(f"Lỗi khi lấy bản đồ keyframe: {e}")
        return jsonify({"error": str(e)}), 500

# (CẬP NHẬT) API MỚI ĐỂ NHẬN CÂU TRẢ LỜI
# (THAY THẾ TOÀN BỘ HÀM CŨ BẰNG HÀM NÀY)
@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    try:
        # 1. (THAY ĐỔI) Lấy wrapper JSON
        wrapper_data = request.get_json()
        if not wrapper_data:
            raise ValueError("Không nhận được dữ liệu JSON.")

        # 2. (THAY ĐỔI) Tách các ID và payload
        evaluation_id = wrapper_data.get('evaluation_id')
        session_id = wrapper_data.get('session_id')
        answer_payload = wrapper_data.get('answer_payload') # Đây là một dict

        if not evaluation_id or not session_id or not answer_payload:
            raise ValueError("Thiếu evaluation_id, session_id, hoặc answer_payload trong request.")

        # 3. In ra màn hình (Gói hàng nhận được)
        print("--- NHẬN ĐƯỢC GÓI HÀNG TỪ JAVASCRIPT ---")
        print(json.dumps(wrapper_data, indent=2, ensure_ascii=False))
        print("------------------------------------------")

        # 4. (THAY ĐỔI) Tạo URL của BTC với ID động
        if not BTC_API_BASE_URL:
            raise ValueError("Biến BTC_API_BASE_URL chưa được thiết lập.")

        btc_url = f"{BTC_API_BASE_URL}/api/v2/submit/{evaluation_id}?session={session_id}"

        print(f"--- ĐANG CHUYỂN TIẾP ĐẾN API CỦA BTC ---")
        print(f"URL: {btc_url}")

        # 5. (THAY ĐỔI) Chuyển đổi payload câu trả lời (dict) thành chuỗi JSON
        # Đây là chuỗi JSON gốc (QA/KIS/TRAKE) mà BTC cần
        answer_string = json.dumps(answer_payload)

        print(f"Payload gửi đi: {answer_string}")

        # 6. Gửi gói hàng (answer_string) đến BTC
        response = requests.post(
            btc_url,
            data=answer_string, # Gửi chuỗi JSON của *câu trả lời*
            headers={ 'Content-Type': 'application/json' }
        )

        response.raise_for_status() # Báo lỗi nếu BTC trả 4xx/5xx

        response_text = response.text
        print(f"--- BTC TRẢ VỀ THÀNH CÔNG ---: {response_text}")

        # 7. Trả kết quả thành công về cho Javascript
        return jsonify({
            "status": "success", 
            "message": "Đã gửi thành công đến BTC.",
            "btc_response": response_text 
        })

    except requests.exceptions.HTTPError as http_err:
        # Lỗi từ server BTC
        response_text = ""
        try:
            # Cố gắng đọc lỗi JSON từ BTC
            response_text = http_err.response.text
        except Exception:
            response_text = "Không thể đọc phản hồi lỗi từ BTC."

        print(f"Lỗi HTTP từ BTC: {http_err.response.status_code}\n{response_text}")
        return jsonify({
            "status": "error", 
            "message": f"Lỗi từ server BTC ({http_err.response.status_code})",
            "btc_response": response_text
        }), 500

    except Exception as e:
        # Lỗi chung (ví dụ: thiếu ID,...)
        print(f"Lỗi khi xử lý /submit_answer: {e}")
        return jsonify({"status": "error", "message": str(e), "btc_response": None}), 500


# --- VÒNG SƠ TUYỂN AIC26: resolve frame thật và xuất submission.zip ---
SUBMISSION_QUERY_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}-(kis|qa|trake)$",
    re.IGNORECASE,
)


def _submission_query_name(raw_name):
    name = str(raw_name or "").strip()
    if name.lower().endswith((".txt", ".csv")):
        name = name.rsplit(".", 1)[0]
    match = SUBMISSION_QUERY_ID_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(
            f"Tên query không hợp lệ: {name!r}; tên phải kết thúc bằng -kis, -qa hoặc -trake."
        )
    return name, match.group(1).lower()


@app.route('/submission/resolve_candidates', methods=['POST'])
def resolve_submission_candidates():
    """Map kết quả retrieval về frame_idx thật trước khi đưa vào CSV."""
    try:
        payload = request.get_json() or {}
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("candidates phải là một JSON array.")
        if len(candidates) > 1000:
            raise ValueError("Chỉ resolve tối đa 1.000 candidates mỗi request.")

        resolved = []
        errors = []
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                errors.append({"index": position, "error": "Candidate không phải object."})
                continue

            video_id = str(
                candidate.get("videoId") or candidate.get("video_id") or ""
            ).upper()
            frame_n = candidate.get("frame_n")
            path = candidate.get("path") or candidate.get("web_path")

            if path:
                web_path, parsed_video_id, parsed_frame_n = get_web_path(path)
                if web_path:
                    path = web_path
                if parsed_video_id and parsed_video_id != "N/A":
                    video_id = parsed_video_id
                if parsed_frame_n is not None:
                    frame_n = int(parsed_frame_n)

            meta = None
            if video_id in metadata_cache and frame_n is not None:
                meta = metadata_cache[video_id].get(int(frame_n))

            frame_idx = meta.get("frame_idx") if meta else candidate.get("frame_idx")
            if not video_id or frame_idx is None:
                errors.append({
                    "index": position,
                    "error": "Không map được video_id/frame_idx.",
                })
                continue

            resolved.append({
                "videoId": video_id,
                "frameIdx": int(frame_idx),
                "path": (meta or {}).get("path") or path,
                "score": float(candidate.get("score", 0) or 0),
            })

        return jsonify({"resolved": resolved, "errors": errors})
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/submission/playback', methods=['POST'])
def resolve_submission_playback():
    """Return the video URL nearest to a submitted frame_idx."""
    try:
        payload = request.get_json() or {}
        video_id = str(payload.get("videoId") or "").upper()
        frame_idx = int(payload.get("frameIdx"))
        video_records = metadata_cache.get(video_id)
        if not video_records:
            raise ValueError(f"Không tìm thấy video {video_id!r}.")

        closest = min(
            video_records.values(),
            key=lambda record: abs(int(record.get("frame_idx", 0)) - frame_idx),
        )
        pts_time = float(closest.get("pts_time", 0) or 0)
        watch_url = video_url_cache.get(video_id)
        if watch_url:
            separator = '&' if '?' in watch_url else '?'
            playback_url = f"{watch_url}{separator}t={int(pts_time)}s"
        else:
            playback_url = None

        return jsonify({
            "videoId": video_id,
            "requestedFrameIdx": frame_idx,
            "frameIdx": int(closest.get("frame_idx", 0)),
            "pts_time": pts_time,
            "path": closest.get("path"),
            "playback_url": playback_url,
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/submission/export', methods=['POST'])
def export_preliminary_submission():
    """Validate ordered rows and return a ZIP containing submission/*.csv."""
    try:
        payload = request.get_json() or {}
        queries = payload.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError("Chưa có query nào để xuất.")

        archive_buffer = io.BytesIO()
        used_names = set()
        with zipfile.ZipFile(
            archive_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for query in queries:
                if not isinstance(query, dict):
                    raise ValueError("Mỗi query phải là một object.")
                query_name, inferred_type = _submission_query_name(query.get("id"))
                query_type = str(query.get("type") or inferred_type).strip().lower()
                if query_type != inferred_type:
                    raise ValueError(
                        f"Loại {query_type!r} không khớp tên file {query_name!r}."
                    )
                normalized_name = query_name.lower()
                if normalized_name in used_names:
                    raise ValueError(f"Trùng query: {query_name}.")
                used_names.add(normalized_name)

                rows = query.get("rows")
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"{query_name} chưa có kết quả.")
                if len(rows) > 100:
                    raise ValueError(f"{query_name} vượt quá 100 dòng.")

                event_count = int(query.get("eventCount") or 0)
                output = io.StringIO(newline="")
                writer = csv.writer(output, lineterminator="\n")
                seen_rows = set()
                for row_number, row in enumerate(rows, start=1):
                    if not isinstance(row, dict):
                        raise ValueError(f"{query_name} dòng {row_number} không hợp lệ.")
                    video_id = str(row.get("videoId") or "").upper()
                    if not re.fullmatch(r"L\d{2}_V\d+", video_id):
                        raise ValueError(
                            f"{query_name} dòng {row_number}: video ID không hợp lệ."
                        )

                    if query_type in {"kis", "qa"}:
                        frame_idx = int(row.get("frameIdx"))
                        csv_row = [video_id, frame_idx]
                        if query_type == "qa":
                            answer = str(row.get("answer") or "")
                            if not answer:
                                raise ValueError(
                                    f"{query_name} dòng {row_number}: thiếu answer."
                                )
                            if len(answer) > 100:
                                raise ValueError(
                                    f"{query_name} dòng {row_number}: answer vượt 100 ký tự."
                                )
                            csv_row.append(answer)
                    else:
                        frame_indices = row.get("frameIndices")
                        if not isinstance(frame_indices, list):
                            raise ValueError(
                                f"{query_name} dòng {row_number}: thiếu danh sách frame TRAKE."
                            )
                        frame_indices = [int(value) for value in frame_indices]
                        if event_count < 2 or len(frame_indices) != event_count:
                            raise ValueError(
                                f"{query_name} dòng {row_number}: cần đúng {event_count} frames."
                            )
                        if any(
                            current <= previous
                            for previous, current in zip(frame_indices, frame_indices[1:])
                        ):
                            raise ValueError(
                                f"{query_name} dòng {row_number}: frames phải tăng theo thời gian."
                            )
                        csv_row = [video_id, *frame_indices]

                    row_key = tuple(csv_row)
                    if row_key in seen_rows:
                        raise ValueError(f"{query_name} có dòng trùng: {csv_row}.")
                    seen_rows.add(row_key)
                    writer.writerow(csv_row)

                archive.writestr(
                    f"submission/{query_name}.csv",
                    output.getvalue().encode("utf-8"),
                )

        archive_buffer.seek(0)
        return send_file(
            archive_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name="submission.zip",
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

# --- Các hàm phục vụ file tĩnh ---
PUBLIC_STATIC_FILES = {
    "style.css",
    "script.js",
    "share_client.css",
    "share_client.js",
    "logo_wud.jpg",
    "submission-builder.css",
    "submission-builder.js",
    "submission-store.js",
}


@app.route('/health', methods=['GET'])
def health():
    """Lightweight readiness check; does not force the lazy Jina model to load."""
    jina_available, jina_reason = jina_text_encoder.availability()
    hybrid_available = jina_available and jina_caption_index is not None
    return jsonify({
        "status": "ok" if hybrid_available else "degraded",
        "device": device,
        "records": len(image_records),
        "jina": {"available": jina_available, "reason": jina_reason},
        "jina_hybrid": {
            "available": hybrid_available,
            "reason": jina_caption_index_reason if jina_available else jina_reason,
        },
        "ocr": {"available": bm25_ocr_index is not None},
        "asr_for_fusion": {"available": bm25_asr_index is not None},
        "auto_crop": {"available": yolo_model is not None},
    })


@app.route('/')
def serve_index(): return send_from_directory(str(BASE_DIR), 'index.html')
@app.route('/submission-builder')
def serve_submission_builder():
    return send_from_directory(str(BASE_DIR), 'submission-builder.html')
@app.route('/<path:path>')
def serve_static(path):
    if path not in PUBLIC_STATIC_FILES:
        abort(404)
    return send_from_directory(str(BASE_DIR), path)
@app.route('/Keyframes/<path:path>')
def serve_keyframes(path): return send_from_directory(str(KEYFRAMES_DIR), path)

# --- CHẠY APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
