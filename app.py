# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
#BTC_EVALUATION_ID = "dda49193-bcb6-4e7d-880f-bf7ec60046ee"
#BTC_SESSION_ID = "tlMIiLdLV-yTB_ENJx6gDtimFMNYL5qk"
BTC_API_BASE_URL = "https://eventretrieval.oj.io.vn"
import torch
import requests
import numpy as np
import json
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_from_directory, g
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
import faiss 
from pathlib import Path
import gc
import os
import pandas as pd
from test_reranker import BlipReranker # (THÊM MỚI) Import Reranker
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
MODEL_NAME = "apple/DFN5B-CLIP-ViT-H-14-378"
ADAPTER_PATH = "fine_tuned_model_lora_2025"
EMBEDDINGS_PATH = "image_embeddings.npy"
PATHS_LIST_PATH = "image_paths.json"
KEYFRAMES_DIR = "Keyframes"
MAP_KEYFRAMES_DIR = "map-keyframes"
MEDIA_INFO_DIR = "media-info"
OCR_DATA_PATH = "ocr.json"
ASR_DATA_DIR = "asr_result"

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
        scores = np.zeros(self.doc_count)
        for term in query:
            if term not in self.idf:
                continue
            term_freqs = []
            for doc in self.corpus:
                term_freqs.append(doc.count(term))
            term_freqs = np.array(term_freqs)
            numerator = term_freqs * (self.k1 + 1)
            denominator = term_freqs + self.k1 * (1 - self.b + self.b * (np.array(self.doc_len) / self.avgdl))
            scores += self.idf[term] * (numerator / denominator)
        return scores
# --- KẾT THÚC CLASS BM25 ---

# --- 4. CẤU HÌNH GROQ API ---
GROQ_API_KEY = "tu_thay_vo_di_ba"  # (TẠM - ĐANG TEST) Dán API key Groq thật vào đây
GROQ_MODEL = "openai/gpt-oss-120b"
try:
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        raise ValueError("Chưa điền GROQ_API_KEY thật vào biến ở trên.")
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

# Tải CLIP model
base_model = CLIPModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
if not Path(ADAPTER_PATH).is_dir():
    raise FileNotFoundError(f"LỖI: Không tìm thấy thư mục adapter LoRA '{ADAPTER_PATH}'.")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to(device)
model.eval()
print(f"Model CLIP đã được tải thành công lên thiết bị: {device.upper()}")

# (THÊM MỚI) Tải mô hình Reranker
print("Đang tải mô hình Reranker (BLIP-ITM)...")
try:
    reranker = BlipReranker(device=device)
except Exception as e:
    print(f"Lỗi khi tải Reranker: {e}")
    reranker = None

# (THÊM MỚI) Tải YOLOv8n model cho auto-crop pre-processing
yolo_model = None
try:
    yolo_model = YOLO("yolov8n.pt")  # Tự động tải về nếu chưa có
    yolo_model.to(device)
    print(f"Model YOLOv8n đã được tải thành công lên thiết bị: {device.upper()}")
except Exception as e:
    print(f"CẢNH BÁO: Không thể tải YOLOv8n model: {e}. Tính năng auto_crop sẽ bị vô hiệu hóa.")
    yolo_model = None

# Tải Semantic data
image_embeddings = np.load(EMBEDDINGS_PATH).astype('float32')
with open(PATHS_LIST_PATH, 'r') as f: image_paths = json.load(f)
print(f"Đã tải {len(image_paths)} vector đặc trưng của ảnh.")

# (THÊM MỚI) Tạo map từ video_id -> prefix đường dẫn
video_id_to_path_prefix = {}
for p_str in image_paths:
    # (SỬA LỖI) Xóa check "Keyframes/" và xử lý cả 2 trường hợp
    # (Trường hợp 1: D:\AIC2025\Keyframes\Keyframes_L21\L21_V001\001.jpg)
    # (Trường hợp 2: Keyframes_L21\L21_V001\001.jpg)
    
    # Chuẩn hóa đường dẫn
    path_str_norm = p_str.replace('\\', '/')
    
    # Tìm vị trí của "Keyframes_" (ví dụ: Keyframes_L21)
    kf_index = path_str_norm.find("Keyframes_")
    
    if kf_index != -1:
        # Tách phần sau "Keyframes/" (ví dụ: Keyframes_L21/L21_V001/001.jpg)
        sub_path_str = path_str_norm[kf_index:]
        p = Path(sub_path_str)
        if len(p.parts) > 1:
            # p.parts[0] là 'Keyframes_L21', p.parts[1] là 'L21_V001'
            video_id = p.parts[1]
            if video_id not in video_id_to_path_prefix:
                # Lưu prefix: Keyframes_L21/L21_V001
                video_id_to_path_prefix[video_id] = str(Path(p.parts[0]) / p.parts[1]).replace('\\', '/')
print(f"Đã tạo map prefix cho {len(video_id_to_path_prefix)} video.")


# Tải OCR data
try:
    with open(OCR_DATA_PATH, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    print(f"Đã tải {len(ocr_data)} bản ghi OCR.")
except Exception as e:
    print(f"Lỗi khi tải file OCR '{OCR_DATA_PATH}': {e}")
    ocr_data = []

# (CẬP NHẬT) Tải ASR data
asr_data = [] 
asr_corpus_tokenized = [] 
asr_video_map = {} 
print("Đang tải dữ liệu ASR...")
asr_dir_path = Path(ASR_DATA_DIR)
if not asr_dir_path.is_dir():
    print(f"CẢNH BÁO: Không tìm thấy thư mục '{ASR_DATA_DIR}'. Bỏ qua tìm kiếm ASR.")
else:
    for json_file in asr_dir_path.glob("**/*.json"):
        video_id = json_file.stem
        video_segments = [] 
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # (SỬA LỖI) Dùng index của list *đã lọc*
                filtered_index = 0 
                for segment in data.get('segments', []): 
                    text = segment.get('text', '').strip()
                    if text:
                        segment_data = {
                            "video_id": video_id,
                            "text": text,
                            "start": segment.get('start', 0),
                            "end": segment.get('end', 0),
                            # (SỬA LỖI) Dùng index đã lọc, không dùng index gốc
                            "video_segment_index": filtered_index 
                        }
                        asr_data.append(segment_data) 
                        video_segments.append(segment_data) 
                        asr_corpus_tokenized.append(text.lower().split())
                        filtered_index += 1 # (SỬA LỖI) Tăng index đã lọc
                        
            asr_video_map[video_id] = video_segments 
        except Exception as e:
            print(f"Lỗi khi đọc file ASR {json_file}: {e}")
print(f"Đã tải {len(asr_data)} phân đoạn ASR từ {len(asr_video_map)} video.")


# Tải Metadata
metadata_cache = {} 
keyframe_time_cache = {} 
print("Đang tải metadata từ 'map-keyframes'...")
for csv_file in Path(MAP_KEYFRAMES_DIR).glob("**/*.csv"):
    video_id = csv_file.stem
    try:
        df = pd.read_csv(csv_file)
        metadata_cache[video_id] = df.set_index('n').to_dict('index')
        df['pts_time'] = pd.to_numeric(df['pts_time'], errors='coerce')
        df = df.dropna(subset=['pts_time'])
        video_fps = None
        if not df.empty and 'fps' in df.columns:
            video_fps = float(df['fps'].iloc[0])
        df = df.sort_values('pts_time')
        times_list = df['pts_time'].tolist()
        # Đảm bảo data_list là list các tuple [frame_n, frame_idx]
        data_list = list(zip(df['n'].tolist(), df['frame_idx'].tolist()))
        keyframe_time_cache[video_id] = {
            "times": times_list, 
            "data": data_list,
            "fps": video_fps     
        }
    except Exception as e:
        print(f"Lỗi khi xử lý file metadata {csv_file}: {e}")
print("Tải metadata và index thời gian keyframe hoàn tất.")


# Tải Media Info
media_info_cache = {}
print("Đang tải thông tin media từ 'media-info'...")
for json_file in Path(MEDIA_INFO_DIR).glob("**/*.json"):
    video_id = json_file.stem
    with open(json_file, 'r', encoding='utf-8') as f:
        media_info_cache[video_id] = json.load(f)
print("Tải media info hoàn tất.")

# --- 6. XÂY DỰNG CÁC INDEX TÌM KIẾM ---

# Index 1: FAISS
print("Đang xây dựng index tìm kiếm với FAISS...")
faiss.normalize_L2(image_embeddings)
index = faiss.IndexFlatIP(image_embeddings.shape[1])
index.add(image_embeddings)
print("Xây dựng index FAISS hoàn tất!")


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
# --- KẾT THÚC HÀM ---


# Index 2: BM25 cho OCR
print("Đang xây dựng index tìm kiếm với BM25 (cho OCR)...")
bm25_ocr_index = None
if ocr_data:
    tokenized_corpus_ocr = []
    print("Bắt đầu làm sạch dữ liệu OCR...")
    for item in ocr_data:
        original_text = item.get('ocr_text', '')
        cleaned_text = clean_ocr_text(original_text) 
        tokenized_corpus_ocr.append(cleaned_text.split())
    print("Làm sạch OCR hoàn tất. Đang huấn luyện BM25...")
    bm25_ocr_index = BM25(tokenized_corpus_ocr, k1=1.5, b=0.75) 
    print(f"Xây dựng index BM25 (OCR) (tự implement) hoàn tất cho {len(tokenized_corpus_ocr)} văn bản.")
else:
    print("Không có dữ liệu OCR để xây dựng index BM25.")


# Index 3: BM25 cho ASR
print("Đang xây dựng index tìm kiếm với BM25 (cho ASR)...")
bm25_asr_index = None
if asr_corpus_tokenized:
    bm25_asr_index = BM25(asr_corpus_tokenized, k1=1.5, b=0.75) 
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

def translate_to_english(text):
    if not groq_client: return text
    try:
        prompt = f"""Translate the following Vietnamese text to English.
        Only output the translated text, nothing else.
        Vietnamese text: "{text}"
        English translation:"""
        translated_text = groq_generate(prompt).strip()
        print(f"Đã dịch '{text}' -> '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"Lỗi khi dịch bằng Groq: {e}"); return text

# --- 8. CÁC API ---

# (XÓA BỎ) Hàm helper get_request_data()
# def get_request_data(): ...

# (CẬP NHẬT) Hàm chuẩn hóa đường dẫn web
def get_web_path(original_path):
    path_str = original_path.replace('\\', '/')
    kf_index = path_str.find("Keyframes_")
    if kf_index != -1:
        sub_path = path_str[kf_index:] # VD: Keyframes_L21/L21_V001/001.jpg
        web_path = "Keyframes/" + sub_path
        
        p = Path(sub_path)
        video_id = p.parts[1] if len(p.parts) > 1 else "N/A"
        frame_n_str = p.stem
        
        # (SỬA LỖI) Chỉ trả về frame_n nếu nó là số
        if frame_n_str.isdigit():
            return web_path, video_id, frame_n_str

    return None, "N/A", None # Trả về None cho frame_n nếu không hợp lệ

# === (THÊM MỚI) TẢI BEIT3 EMBEDDINGS + MODEL CHO FUSION SIMILAR SEARCH ===
# BEIT3 (fine-grained, chỉ có vision tower - không có text tower) dùng để
# kết hợp (fusion) với CLIP khi tìm ảnh tương tự bằng ảnh mẫu (Similar Search).
BEIT3_TIMM_NAME = "beit3_base_patch16_224.in22k_ft_in1k"
BEIT3_VECTORS_DIR = "beit3_vectors/vectors_beit3"

beit3_embeddings = None          # (N_beit3, 768) đã L2-normalize
beit3_web_path_to_idx = {}       # web_path (giống format image_paths) -> row trong beit3_embeddings
beit3_model = None
beit3_transform = None

print("Đang tải BEIT3 embeddings (fine-grained, cho fusion Similar Search)...")
try:
    beit3_dir = Path(BEIT3_VECTORS_DIR)
    emb_chunks, all_beit3_paths = [], []
    for emb_file in sorted(beit3_dir.glob("*_beit3_embeddings.npy")):
        paths_file = emb_file.with_name(emb_file.name.replace("_embeddings.npy", "_paths.json"))
        if not paths_file.exists():
            continue
        emb_chunks.append(np.load(emb_file).astype('float32'))
        with open(paths_file, 'r', encoding='utf-8') as f:
            all_beit3_paths.extend(json.load(f))
    if emb_chunks:
        beit3_embeddings = np.concatenate(emb_chunks, axis=0)
        faiss.normalize_L2(beit3_embeddings)
        for row_idx, p_str in enumerate(all_beit3_paths):
            web_path, _, _ = get_web_path(p_str)
            if web_path:
                beit3_web_path_to_idx[web_path] = row_idx
        print(f"Đã tải {beit3_embeddings.shape[0]} vector BEIT3, khớp {len(beit3_web_path_to_idx)} frame với CLIP.")
    else:
        print(f"CẢNH BÁO: Không tìm thấy shard BEIT3 nào trong '{BEIT3_VECTORS_DIR}'.")
except Exception as e:
    print(f"Lỗi khi tải BEIT3 embeddings: {e}")
    beit3_embeddings = None

try:
    import timm
    beit3_model = timm.create_model(BEIT3_TIMM_NAME, pretrained=True, num_classes=0)
    beit3_model.eval().to(device)
    beit3_data_cfg = timm.data.resolve_data_config({}, model=beit3_model)
    beit3_transform = timm.data.create_transform(**beit3_data_cfg)
    print(f"Model BEIT3 (timm) đã tải thành công lên {device.upper()}.")
except Exception as e:
    print(f"CẢNH BÁO: Không thể tải model BEIT3 ({e}). Similar Search sẽ chỉ dùng 100% CLIP.")
    beit3_model = None
    beit3_transform = None

def encode_image_beit3(pil_image):
    """Encode 1 ảnh PIL thành vector BEIT3 768-chiều đã L2-normalize. Trả None nếu model chưa sẵn sàng."""
    if beit3_model is None or beit3_transform is None:
        return None
    try:
        with torch.no_grad():
            tensor = beit3_transform(pil_image).unsqueeze(0).to(device)
            feat = beit3_model(tensor)
        vec = feat.detach().cpu().numpy().astype('float32')
        faiss.normalize_L2(vec)
        return vec[0]
    except Exception as e:
        print(f"Lỗi khi encode ảnh bằng BEIT3: {e}")
        return None

def fuse_clip_beit3_scores(candidate_indices, candidate_clip_sims, beit3_query_vec, alpha):
    """
    Kết hợp điểm CLIP (ngữ nghĩa) và BEIT3 (chi tiết/fine-grained) theo trọng số alpha.
    alpha=1.0 -> 100% CLIP, alpha=0.0 -> 100% BEIT3. Ảnh không có vector BEIT3 (hiếm) coi beit3_sim=0.
    Trả về list (idx, fused_score) đã sort giảm dần.
    """
    fused = []
    for idx, clip_sim in zip(candidate_indices, candidate_clip_sims):
        beit3_sim = 0.0
        if beit3_query_vec is not None and beit3_embeddings is not None:
            web_path, _, _ = get_web_path(image_paths[idx])
            b_idx = beit3_web_path_to_idx.get(web_path) if web_path else None
            if b_idx is not None:
                beit3_sim = float(np.dot(beit3_embeddings[b_idx], beit3_query_vec))
        fused.append((int(idx), alpha * float(clip_sim) + (1.0 - alpha) * beit3_sim))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
# === (KẾT THÚC) BEIT3 FUSION ===

# === (THÊM MỚI) QUERY EXPANSION (Groq) - Theo "[AIC2026] - Query expansion.docx", PLAN A ===
QUERY_EXPANSION_PROMPT_TEMPLATE = """Bạn là một chuyên gia tối ưu hóa tìm kiếm video chuyên nghiệp. Nhiệm vụ của bạn là mở rộng câu truy vấn gốc của người dùng thành 3 khía cạnh ngữ nghĩa khác nhau: mở rộng theo kiểu sát nghĩa; mở rộng nhấn mạnh vai trò; mở rộng nhấn mạnh nơi chốn. Việc này giúp bộ mã hóa của hệ thống dễ dàng so khớp không gian vector với các keyframes của video.

Ví dụ, câu truy vấn gốc "cô gái nấu ăn" được mở rộng thành:
- Sát nghĩa: "con gái nấu"
- Nhấn mạnh vai trò: "nữ đầu bếp đang chuẩn bị món ăn"
- Nhấn mạnh nơi chốn: "cô gái đang nấu nướng ở nhà bếp"

Câu truy vấn gốc cần mở rộng: "{query}"

Chỉ trả lời bằng một object JSON hợp lệ duy nhất, không kèm giải thích, chú thích hay markdown, đúng định dạng:
{{"paraphrase": "...", "role": "...", "location": "..."}}"""

def expand_query_with_groq(query_text):
    """Sinh 3 biến thể ngữ nghĩa (sát nghĩa / vai trò / nơi chốn) từ query gốc bằng Groq."""
    if not groq_client:
        return []
    try:
        prompt = QUERY_EXPANSION_PROMPT_TEMPLATE.format(query=query_text)
        raw = groq_generate(prompt).strip()
        raw = re.sub(r'^```(json)?|```$', '', raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        variants = [parsed.get('paraphrase', ''), parsed.get('role', ''), parsed.get('location', '')]
        variants = [v.strip() for v in variants if v and v.strip()]
        print(f"[QueryExpansion] '{query_text}' -> {variants}")
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
    (dùng cho Fusion search: tỷ lệ CLIP/OCR/ASR do người dùng chỉnh)."""
    if weights is None:
        weights = [1.0] * len(ranked_id_lists)
    scores = {}
    for w, ranked_ids in zip(weights, ranked_id_lists):
        for rank, idx in enumerate(ranked_ids):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
# === (KẾT THÚC) QUERY EXPANSION ===

# === (CẬP NHẬT) OCR/ASR: bỏ hẳn Elasticsearch, dùng thẳng BM25 tự viết (đã build sẵn lúc khởi động) ===
def ocr_candidates(query_text, search_size):
    """Trả list[(score, original_path)] từ bm25_ocr_index (xem class BM25 đầu file)."""
    if not bm25_ocr_index:
        return []
    tokenized_query = clean_ocr_text(query_text).split()
    scores = bm25_ocr_index.get_scores(tokenized_query)
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
    """Trả list[dict{video_id,text,start,end,score}] từ bm25_asr_index."""
    if not bm25_asr_index:
        return []
    tokenized_query = query_text.lower().split()
    scores = bm25_asr_index.get_scores(tokenized_query)
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
             
        query_text = data['query']
        top_k = int(data.get('top_k', 50))
        # (SỬA LỖI) Xử lý 'group' (là boolean true/false)
        group_results = data.get('group', False) 
        should_translate = data.get('translate', False)
        
        # === (LOGIC MỚI) KIỂM TRA XEM QUERY CÓ PHẢI LÀ VIDEO ID KHÔNG ===
        
        # Chuẩn hóa query (ví dụ: " l22_v002 " -> "L22_V002")
        video_id_query = query_text.strip().upper() 
        
        # Kiểm tra xem query này có nằm trong danh sách video ID ta có không
        # (video_id_to_path_prefix được tạo khi khởi động server)
        if video_id_query in video_id_to_path_prefix:
            
            print(f"Phát hiện tìm kiếm theo Video ID: {video_id_query}")
            results = []
            summary = {}
            
            # Lấy prefix đường dẫn, ví dụ: "Keyframes_L22/L22_V002"
            path_prefix = video_id_to_path_prefix[video_id_query] 
            
            # Lấy tất cả metadata cho video này
            video_meta = metadata_cache.get(video_id_query, {})
            
            if not video_meta:
                return jsonify({"results": [], "summary": {}, "error": f"Không tìm thấy metadata cho video {video_id_query}"})

            # Lặp qua tất cả các frame (n) trong metadata của video đó
            for frame_n_int, meta in video_meta.items():
                # Tạo web_path
                # Ví dụ: "Keyframes/Keyframes_L22/L22_V002/001.jpg"
                frame_n_str = str(frame_n_int).zfill(3)
                web_path = f"Keyframes/{path_prefix}/{frame_n_str}.jpg"
                
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0
                
                results.append({
                    "path": web_path, 
                    "videoId": video_id_query, 
                    "score": float(pts_time), # Dùng pts_time làm score để sort
                    "pts_time": float(pts_time)
                })

            summary[video_id_query] = len(results)
            
            # Sắp xếp theo thời gian
            final_results = sorted(results, key=lambda x: x['pts_time'])
            
            # Cắt theo top_k (mặc dù ta lấy hết, nhưng vẫn tôn trọng top_k)
            final_results = final_results[:top_k]
            
            # Nếu user có check "Group", thì ta group lại
            if group_results:
                final_grouped_results = {video_id_query: final_results}
                return jsonify({"results": final_grouped_results, "summary": summary})
            else:
                return jsonify({"results": final_results, "summary": summary})
        
        # === (KẾT THÚC LOGIC MỚI) ===
        
        # Nếu không phải là Video ID, chạy logic tìm kiếm semantic CŨ
        print(f"Đang tìm kiếm semantic cho: '{query_text}'")

        if should_translate:
            english_query = translate_to_english(query_text)
        else:
            print("Bỏ qua bước dịch, tìm kiếm trực tiếp.")
            english_query = query_text

        with torch.no_grad():
            inputs = processor(text=[english_query], return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**inputs)

        query_vector = text_features.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_vector)

        pool_k = top_k * 5 if group_results else top_k

        clip_sim_by_idx = {}
        distances, indices = index.search(query_vector, pool_k)
        ordered_indices = [int(i) for i in indices[0]]
        for i, dist in zip(indices[0], distances[0]):
            clip_sim_by_idx[int(i)] = float(dist)

        results = []
        summary = {}

        for i in ordered_indices:
            original_path = image_paths[i]
            web_path, video_id, frame_n_str = get_web_path(original_path)

            # (SỬA LỖI) Thêm check frame_n_str (không phải None)
            if web_path and frame_n_str:
                frame_n_int = int(frame_n_str)
                meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0

                results.append({
                    "path": web_path,
                    "videoId": video_id,
                    "score": clip_sim_by_idx.get(i, 0.0),
                    "pts_time": float(pts_time)
                })
                summary[video_id] = summary.get(video_id, 0) + 1

        sorted_summary = dict(sorted(summary.items(), key=lambda item: item[1], reverse=True))

        # === RERANK BẰNG BLIP-ITM (CHỈ TOP-10 ĐỂ TRÁNH TIMEOUT) ===
        ENABLE_RERANKER = False
        RERANK_TOP_N = 10  # Chỉ rerank 10 ảnh đầu tiên
        if ENABLE_RERANKER and reranker is not None:
            # Sắp xếp theo score ban đầu của FAISS trước
            results_sorted_by_faiss = sorted(results, key=lambda x: x['score'], reverse=True)
            top_n = results_sorted_by_faiss[:RERANK_TOP_N]
            rest = results_sorted_by_faiss[RERANK_TOP_N:]

            print(f"Đang rerank {len(top_n)} ảnh đầu bằng BLIP-ITM...")
            for r in top_n:
                r['image_path'] = r['path']
            reranked_top = reranker.rerank(english_query, top_n, image_base_dir=".")
            for r in reranked_top:
                r['score'] = r.get('itm_score', r['score'])

            # Ghép lại: top-10 đã rerank + phần còn lại giữ nguyên thứ tự FAISS
            results = reranked_top + rest

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
            final_results = results[:top_k] # Đã được Reranker sắp xếp sẵn
            return jsonify({"results": final_results, "summary": sorted_summary})

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
        # (THÊM MỚI) Trọng số fusion CLIP/BEIT3 từ thanh trượt UI (1.0 = 100% CLIP, 0.0 = 100% BEIT3)
        try:
            beit3_alpha = float(data.get('beit3_weight', 1.0))
        except (TypeError, ValueError):
            beit3_alpha = 1.0
        beit3_alpha = max(0.0, min(1.0, beit3_alpha))

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

        # Đưa target_image (gốc hoặc đã crop) vào CLIP
        with torch.no_grad():
            inputs = processor(images=[target_image], return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)

        query_vector = image_features.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_vector)

        # Dọn dẹp sau CLIP inference
        del inputs, image_features
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # (THÊM MỚI) Encode BEIT3 cho ảnh query (chỉ khi slider không ở 100% CLIP)
        beit3_query_vec = None
        if beit3_alpha < 1.0:
            beit3_query_vec = encode_image_beit3(target_image)
            if beit3_query_vec is None:
                print("CẢNH BÁO: Model/vector BEIT3 không sẵn sàng, fallback về 100% CLIP.")
                beit3_alpha = 1.0

        # (THÊM MỚI) Khi có fusion, lấy pool ứng viên rộng hơn từ CLIP rồi rerank lại bằng BEIT3
        if beit3_query_vec is not None:
            pool_k = min(max(top_k * 20, 500), 5000)
        else:
            pool_k = top_k * 5 if group_results else top_k

        distances, indices = index.search(query_vector, pool_k)

        if beit3_query_vec is not None:
            ordered = fuse_clip_beit3_scores(indices[0], distances[0], beit3_query_vec, beit3_alpha)
            # (SỬA LỖI) pool_k dùng cho fusion rerank rộng hơn nhiều so với số kết quả thực trả về
            # (500-5000 ứng viên) -> phải cắt về đúng số lượng cần thiết TRƯỚC khi build summary,
            # nếu không summary sẽ thống kê nhầm theo cả pool ứng viên bị loại chứ không phải kết quả thật.
            needed = top_k * 5 if group_results else top_k
            ordered = ordered[:needed]
        else:
            ordered = [(int(i), float(dist)) for i, dist in zip(indices[0], distances[0])]

        results = []
        summary = {}

        for i, score in ordered:
            original_path = image_paths[i]
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
#                   được giữ nguyên (frame sự kiện 1 < frame sự kiện 2 < ...) và tổng điểm CLIP lớn nhất,
#                   ĐỒNG THỜI phạt nặng khi 2 sự kiện liên tiếp cách nhau quá xa về thời gian.
# Giai đoạn 2 là bài toán quy hoạch động (DP) - xem _align_event_sequence().
TRAKE_DEFAULT_MAX_GAP_SECONDS = 30.0   # trong khoảng này thì không phạt
TRAKE_DEFAULT_GAP_PENALTY = 0.01       # phạt mỗi giây vượt ngưỡng; vượt 30s ~ mất trọn 1 sự kiện khớp


def _align_event_sequence(candidate_frames, frame_times, event_scores, n_events,
                          max_gap_seconds=TRAKE_DEFAULT_MAX_GAP_SECONDS,
                          gap_penalty_per_sec=TRAKE_DEFAULT_GAP_PENALTY):
    """Chọn chuỗi frame TĂNG DẦN (mỗi sự kiện 1 frame), tối đa hoá tổng điểm CLIP TRỪ ĐI tiền phạt
    khoảng cách thời gian giữa 2 sự kiện liên tiếp.

    Một chuỗi hành động (chạy đà -> giậm nhảy -> ...) diễn ra liên tục trong vài chục giây, nên nếu
    chỉ ràng buộc "frame sau > frame trước" thì DP hay ghép các sự kiện cách nhau vài phút - vốn là
    những cảnh không liên quan trong cùng video. Tiền phạt tuyến tính phần vượt quá max_gap_seconds
    khiến các chuỗi rời rạc như vậy tụt hạng, nhưng vẫn không loại hẳn (phòng khi đáp án thật hơi thưa).

    candidate_frames: list frame_n (int) đã sort tăng dần - các keyframe ứng viên của 1 video.
    frame_times:      list pts_time (giây) song song với candidate_frames, cũng tăng dần.
    event_scores:     dict[(event_index, frame_n)] -> điểm CLIP. Thiếu key = sự kiện đó không
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
        should_translate = data.get('translate', True)

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

        # video_event_scores[video_id][(event_index, frame_n)] = điểm CLIP tốt nhất
        video_event_scores = collections.defaultdict(dict)
        # video_matched_events[video_id] = tập các event_index thực sự có hit trong video đó
        video_matched_events = collections.defaultdict(set)

        # Dịch + encode TẤT CẢ sự kiện một lượt, rồi search FAISS theo batch (1 lượt quét index cho
        # cả N sự kiện, nhanh hơn hẳn so với gọi index.search() riêng từng sự kiện).
        event_queries = [translate_to_english(p) if should_translate else p for p in parts]
        for i, q in enumerate(event_queries):
            print(f"  [Sự kiện {i + 1}/{n_events}] '{q}'")

        with torch.no_grad():
            inputs = processor(text=event_queries, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**inputs)
        query_vectors = text_features.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_vectors)

        distances, indices = index.search(query_vectors, pool_k)  # shape: (n_events, pool_k)

        for event_index in range(n_events):
            for idx, dist in zip(indices[event_index], distances[event_index]):
                idx = int(idx)
                if idx < 0:
                    continue
                web_path, video_id, frame_n_str = get_web_path(image_paths[idx])
                if not web_path or not frame_n_str or video_id == "N/A":
                    continue
                frame_n = int(frame_n_str)
                key = (event_index, frame_n)
                score = float(dist)
                scores_of_video = video_event_scores[video_id]
                if score > scores_of_video.get(key, float('-inf')):
                    scores_of_video[key] = score
                video_matched_events[video_id].add(event_index)

        # Giai đoạn 2: căn chỉnh thời gian trong từng video ứng viên
        sequences = []
        for video_id, event_scores in video_event_scores.items():
            if len(video_matched_events[video_id]) < min_events:
                continue
            prefix = video_id_to_path_prefix.get(video_id)
            if not prefix:
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
                    "path": f"Keyframes/{prefix}/{str(frame_n).zfill(3)}.jpg",
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
        
        with torch.no_grad():
            for img_index, file in enumerate(image_files):
                # Xử lý từng ảnh
                if file.filename == '': continue
                image = Image.open(io.BytesIO(file.read())).convert("RGB")
                
                inputs = processor(images=[image], return_tensors="pt").to(device)
                image_features = model.get_image_features(**inputs)
                
                query_vector = image_features.cpu().numpy().astype('float32')
                faiss.normalize_L2(query_vector)
                
                # Tìm n1 frames cho ảnh này
                distances, indices = index.search(query_vector, top_k_per_image)
                
                for i, dist in zip(indices[0], distances[0]):
                    original_path = image_paths[i]
                    web_path, video_id, frame_n_str = get_web_path(original_path)
                    
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        video_meta = metadata_cache.get(video_id, {})
                        
                        # (CẬP NHẬT) Quét cửa sổ +/- window_size
                        for offset in range(-window_size, window_size + 1):
                            neighbor_n = frame_n_int + offset
                            if neighbor_n in video_meta:
                                neighbor_str = str(neighbor_n).zfill(3)
                                frame_key = (video_id, neighbor_str)
                                
                                # Cập nhật điểm cao nhất nếu xuất hiện nhiều lần trong window
                                current_best = frame_image_scores[frame_key].get(img_index, -1000.0)
                                if float(dist) > current_best:
                                    frame_image_scores[frame_key][img_index] = float(dist)
                                
                                if frame_key not in frame_info_cache:
                                    meta = video_meta[neighbor_n]
                                    pts_time = meta.get('pts_time', 0) if meta else 0
                                    
                                    prefix = video_id_to_path_prefix.get(video_id)
                                    if prefix:
                                        neighbor_web_path = f"Keyframes/{prefix}/{neighbor_str}.jpg"
                                    else:
                                        neighbor_web_path = web_path.replace(f"{frame_n_str}.jpg", f"{neighbor_str}.jpg")
                                        
                                    frame_info_cache[frame_key] = {
                                        "path": neighbor_web_path,
                                        "videoId": video_id,
                                        "pts_time": float(pts_time)
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
            
    except Exception as e:
        print(f"Lỗi trong /search_trake_image: {e}")
        return jsonify({"error": str(e)}), 500
# (CẬP NHẬT) API /search_asr - Dùng ElasticSearch thay BM25
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

        # === PIPELINE GIỮA NGUYÊN: stitching + keyframe mapping ===
        final_results = []
        processed_segments_key = set()
        summary = {}
        tokenized_query = query_text.lower().split()

        def contains_keyword(text, query_tokens):
            text_lower = text.lower()
            for token in query_tokens:
                if token in text_lower:
                    return True
            return False

        for doc in top_k_documents:
            video_id = doc['video_id']
            doc_key = f"{video_id}_{doc['start']}"
            if doc_key in processed_segments_key:
                continue

            current_merge_group = [doc]
            processed_segments_key.add(doc_key)
            video_segments_list = asr_video_map.get(video_id, [])

            # Tìm vị trí segment trong danh sách theo start time
            seg_idx = None
            for i, seg in enumerate(video_segments_list):
                if abs(seg.get('start', -1) - doc['start']) < 0.5:
                    seg_idx = i
                    break

            if seg_idx is not None:
                prev_index = seg_idx - 1
                if prev_index >= 0:
                    prev_segment = video_segments_list[prev_index]
                    if contains_keyword(prev_segment['text'], tokenized_query):
                        current_merge_group.insert(0, prev_segment)
                        processed_segments_key.add(f"{video_id}_{prev_segment.get('start','')}")

                next_index = seg_idx + 1
                if next_index < len(video_segments_list):
                    next_segment = video_segments_list[next_index]
                    if contains_keyword(next_segment['text'], tokenized_query):
                        current_merge_group.append(next_segment)
                        processed_segments_key.add(f"{video_id}_{next_segment.get('start','')}")

            if len(current_merge_group) == 1:
                final_doc = doc.copy()
            else:
                final_doc = {
                    "video_id": video_id,
                    "text":     " + ".join([seg['text'] for seg in current_merge_group]),
                    "start":    current_merge_group[0]['start'],
                    "end":      current_merge_group[-1]['end'],
                    "score":    doc['score']
                }

            media_info = media_info_cache.get(video_id, {})
            final_doc['watch_url'] = media_info.get('watch_url')

            target_start_time = final_doc['start']
            closest_frame_data = find_closest_keyframe(video_id, target_start_time)
            final_doc['frame_n']   = closest_frame_data.get('frame_n')
            final_doc['frame_idx'] = closest_frame_data.get('frame_idx')

            final_doc['web_path'] = None
            prefix = video_id_to_path_prefix.get(video_id)
            frame_n = final_doc['frame_n']
            if prefix and frame_n is not None:
                final_doc['web_path'] = f"Keyframes/{prefix}/{str(frame_n).zfill(3)}.jpg"
            
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


# (THÊM MỚI) API /search_fusion - Gộp CLIP (semantic) + OCR + ASR bằng Reciprocal Rank Fusion có trọng số
# Mỗi khoảnh khắc được định danh bằng (video_id, frame_n) để gộp điểm giữa 3 nguồn có thang đo khác nhau
# (CLIP: cosine similarity, OCR/ASR: điểm BM25 - không thể so trực tiếp được).
# 3 ô query riêng (CLIP/OCR/ASR) vì nội dung mong muốn cho từng nguồn thường khác nhau: CLIP cần mô tả
# cảnh bằng tiếng Anh, OCR/ASR cần đúng từ khoá tiếng Việt xuất hiện trên màn hình/lời thoại.
@app.route('/search_fusion', methods=['POST'])
def search_fusion():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request phải là JSON"}), 400

        query_clip = data.get('query_clip', '').strip()
        query_ocr = data.get('query_ocr', '').strip()
        query_asr = data.get('query_asr', '').strip()
        if not query_clip and not query_ocr and not query_asr:
            return jsonify({"results": [], "summary": {}})

        # Trọng số thô (mặc định bằng nhau) - chỉ tỷ lệ tương đối giữa 3 số này mới ảnh hưởng thứ hạng RRF,
        # nên không cần chuẩn hoá về tổng=1 trước khi tính.
        weight_clip = float(data.get('weight_clip', 1.0))
        weight_ocr = float(data.get('weight_ocr', 1.0))
        weight_asr = float(data.get('weight_asr', 1.0))

        top_k = int(data.get('top_k', 50))
        group_results = data.get('group', False)
        should_translate = data.get('translate', True)

        pool_k = max(top_k * 5, 100)

        clip_ranked_keys, ocr_ranked_keys, asr_ranked_keys = [], [], []
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

        # --- 1. Nhánh CLIP (semantic, FAISS) ---
        if query_clip:
            try:
                english_query = translate_to_english(query_clip) if should_translate else query_clip
                with torch.no_grad():
                    inputs = processor(text=[english_query], return_tensors="pt", padding=True, truncation=True).to(device)
                    text_features = model.get_text_features(**inputs)
                query_vector = text_features.cpu().numpy().astype('float32')
                faiss.normalize_L2(query_vector)
                distances, indices = index.search(query_vector, pool_k)
                for i in indices[0]:
                    i = int(i)
                    if i < 0:
                        continue
                    web_path, video_id, frame_n_str = get_web_path(image_paths[i])
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        key = (video_id, frame_n_int)
                        clip_ranked_keys.append(key)
                        meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                        register(key, web_path, video_id, meta.get('pts_time', 0) if meta else 0, "CLIP")
            except Exception as e:
                print(f"[Fusion] Lỗi nhánh CLIP: {e}")

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
                    prefix = video_id_to_path_prefix.get(video_id)
                    if frame_n is None or not prefix:
                        continue
                    frame_n_int = int(frame_n)
                    web_path = f"Keyframes/{prefix}/{str(frame_n_int).zfill(3)}.jpg"
                    key = (video_id, frame_n_int)
                    asr_ranked_keys.append(key)
                    meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                    pts_time = meta.get('pts_time', 0) if meta else src.get("start", 0.0)
                    register(key, web_path, video_id, pts_time, "ASR")
            except Exception as e:
                print(f"[Fusion] Lỗi nhánh ASR: {e}")

        # --- 4. Gộp bằng RRF có trọng số (tỷ lệ CLIP/OCR/ASR do người dùng chỉnh qua slider) ---
        branches = [
            (clip_ranked_keys, weight_clip),
            (ocr_ranked_keys, weight_ocr),
            (asr_ranked_keys, weight_asr),
        ]
        branches = [(lst, w) for lst, w in branches if lst]
        if not branches:
            return jsonify({"results": [], "summary": {}, "error": "Không có nhánh nào (CLIP/OCR/ASR) trả về kết quả."})

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
                "matched_by": sorted(key_sources[key])  # ví dụ ["CLIP", "OCR"] - để UI hiện badge nguồn khớp
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
        # (SỬA LỖI) Xử lý đường dẫn đến từ web (đã có Keyframes/)
        p = Path(image_path.replace("Keyframes/", "")) # Bỏ prefix
        # Giả định cấu trúc: Keyframes_XXX/VIDEO_ID/frame.jpg
        video_id = p.parts[-2]
        frame_n = int(p.stem)
        meta = metadata_cache.get(video_id, {}).get(frame_n, {})
        media_info = media_info_cache.get(video_id, {})
        meta['n'] = frame_n
        watch_url = media_info.get('watch_url')
        if watch_url and 'pts_time' in meta and meta['pts_time'] is not None:
            seconds = int(float(meta['pts_time']))
            meta['playback_url'] = f"{watch_url}&t={seconds}s"
        else:
            meta['playback_url'] = watch_url
        return jsonify(meta)
    except Exception as e: return jsonify({"error": str(e)}), 500

# (CẬP NHẬT) API /neighbor_frames
@app.route('/neighbor_frames', methods=['POST'])
def get_neighbor_frames():
    try:
        image_path = request.json['image_path']
        p = Path(image_path.replace("Keyframes/", "")) # Bỏ prefix
        video_id = p.parts[-2]
        frame_n = int(p.stem)
        video_frames = sorted(metadata_cache.get(video_id, {}).keys())
        if not video_frames: return jsonify({"neighbors": []})
        current_index = video_frames.index(frame_n)
        start = max(0, current_index - 5)
        end = min(len(video_frames), current_index + 6)
        neighbor_ns = video_frames[start:end]
        parent_folder = p.parts[-3] 
        neighbors = [f"Keyframes/{parent_folder}/{video_id}/{str(n).zfill(3)}.jpg" for n in neighbor_ns]
        return jsonify({"neighbors": neighbors})
    except Exception as e: return jsonify({"error": str(e)}), 500

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

# --- Các hàm phục vụ file tĩnh ---
@app.route('/')
def serve_index(): return send_from_directory('.', 'index.html')
@app.route('/<path:path>')
def serve_static(path): return send_from_directory('.', path)
@app.route('/Keyframes/<path:path>')
def serve_keyframes(path): return send_from_directory(KEYFRAMES_DIR, path)

# --- CHẠY APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

