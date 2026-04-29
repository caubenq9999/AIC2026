# --- 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
#BTC_EVALUATION_ID = "dda49193-bcb6-4e7d-880f-bf7ec60046ee"
#BTC_SESSION_ID = "tlMIiLdLV-yTB_ENJx6gDtimFMNYL5qk"
BTC_API_BASE_URL = "https://eventretrieval.oj.io.vn"
import torch
import requests
import numpy as np
import json
from flask import Flask, request, jsonify, send_from_directory, g
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
import faiss 
from pathlib import Path
import gc
import os
import pandas as pd
import google.genai as genai
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

# --- 4. CẤU HÌNH GEMINI API ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY: raise ValueError("Biến môi trường GEMINI_API_KEY chưa được thiết lập.")
    genai.configure(api_key=GEMINI_API_KEY)
    translation_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Kết nối với Gemini API thành công.")
except Exception as e:
    print(f"Lỗi khi cấu hình Gemini API: {e}")
    translation_model = None

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
CORS(app)

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
    if not translation_model: return text
    try:
        prompt = f"""Translate the following Vietnamese text to English. 
        Only output the translated text, nothing else. 
        Vietnamese text: "{text}"
        English translation:""" 
        response = translation_model.generate_content(prompt)
        translated_text = response.text.strip()
        print(f"Đã dịch '{text}' -> '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"Lỗi khi dịch bằng Gemini: {e}"); return text

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
        english_query = query_text
        if should_translate:
            english_query = translate_to_english(query_text)
        else:
            print("Bỏ qua bước dịch, tìm kiếm trực tiếp.")
        
        with torch.no_grad():
            inputs = processor(text=[english_query], return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**inputs)
        
        query_vector = text_features.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = index.search(query_vector, top_k * 5 if group_results else top_k) 
        
        results = []
        summary = {}
        
        for i, dist in zip(indices[0], distances[0]):
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
                    "score": float(dist),
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

        distances, indices = index.search(query_vector, top_k * 5 if group_results else top_k)

        results = []
        summary = {}

        for i, dist in zip(indices[0], distances[0]):
            original_path = image_paths[i]
            web_path, video_id, frame_n_str = get_web_path(original_path)

            if web_path and frame_n_str:
                frame_n_int = int(frame_n_str)
                meta = metadata_cache.get(video_id, {}).get(frame_n_int, {})
                pts_time = meta.get('pts_time', 0) if meta and meta.get('pts_time') else 0

                results.append({
                    "path": web_path,
                    "videoId": video_id,
                    "score": float(dist),
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
        
        if not bm25_ocr_index: 
            return jsonify({"error": "Index BM25 (OCR) chưa được khởi tạo."}), 500
        if not query_text:
            return jsonify({"results": [], "summary": {}})
        
        tokenized_query = query_text.split()
        scores = bm25_ocr_index.get_scores(tokenized_query)
        
        top_k_indices = np.argsort(scores)[::-1][:top_k * 5 if group_results else top_k] 
        
        results = []
        summary = {}
        
        for i in top_k_indices:
            score = scores[i]
            if score <= 0: continue
            
            item = ocr_data[i]
            original_path = item.get('path', '')
            if not original_path: continue
            
            web_path, video_id, frame_n_str = get_web_path(original_path)

            # (SỬA LỖI) Thêm check frame_n_str (không phải None)
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

@app.route('/search_trake_02', methods=['POST'])
def search_trake_02():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Request phải là JSON"}), 400
            
        query_text = data.get('query', '')
        top_k_final = int(data.get('top_k', 50))
        group_results = data.get('group', False)
        should_translate = data.get('translate', True)
        
        # 1. Tách query thành các parts
        parts = [p.strip() for p in query_text.split(';') if p.strip()]
        
        if not parts:
            return jsonify({"error": "Query rỗng hoặc thiếu dấu ';' để tách các phần"}), 400
        if len(parts) < 2:
            return jsonify({"error": "TRAKE.02 yêu cầu ít nhất 2 phần (phân tách bằng ';')"}), 400

        # (CẬP NHẬT) Ngưỡng "common points" (cho phép thiếu 1 part nếu có nhiều part)
        min_common = max(2, len(parts) - 1) if len(parts) >= 3 else len(parts)
        
        # (CẬP NHẬT) Nới khung hình (Frame Windowing)
        window_size = int(data.get('window_size', 5)) # Mặc định nới +/- 5 frames
        
        # n1 (số frame cần lấy cho mỗi part)
        # Tăng mạnh số lượng lấy từ index (ví dụ 10000) để đảm bảo có giao nhau
        top_k_per_part = max(top_k_final * 100, 10000)

        print(f"[TRAKE.02] Đang tìm kiếm {len(parts)} phần, n1={top_k_per_part}, min_common={min_common}, window_size=±{window_size}")

        # Store frame occurrences
        # Key: (video_id, frame_n_str), Value: Dict[part_index -> max_score]
        frame_part_scores = collections.defaultdict(lambda: collections.defaultdict(float))
        # Key: (video_id, frame_n_str), Value: {path, videoId, pts_time}
        frame_info_cache = {} 
        # Key: video_id, Value: count (for summary)
        summary_counter = collections.defaultdict(int)

        with torch.no_grad():
            for part_index, part_text in enumerate(parts):
                # 3. Search for each part
                english_query = part_text
                if should_translate:
                    english_query = translate_to_english(part_text)
                
                print(f"  [Part {part_index}]: '{english_query}'")
                inputs = processor(text=[english_query], return_tensors="pt", padding=True, truncation=True).to(device)
                text_features = model.get_text_features(**inputs)
                query_vector = text_features.cpu().numpy().astype('float32')
                faiss.normalize_L2(query_vector)
                
                # Tìm n1 (top_k_per_part) frame cho part này
                distances, indices = index.search(query_vector, top_k_per_part) 
                
                for i, dist in zip(indices[0], distances[0]):
                    original_path = image_paths[i]
                    web_path, video_id, frame_n_str = get_web_path(original_path)
                    
                    if web_path and frame_n_str:
                        frame_n_int = int(frame_n_str)
                        # Lấy danh sách frame thực tế của video này để tránh nới ra frame không tồn tại
                        video_meta = metadata_cache.get(video_id, {})
                        
                        # (CẬP NHẬT) Quét cửa sổ +/- window_size
                        for offset in range(-window_size, window_size + 1):
                            neighbor_n = frame_n_int + offset
                            if neighbor_n in video_meta: # Chỉ nới tới những frame có thật
                                neighbor_str = str(neighbor_n).zfill(3)
                                frame_key = (video_id, neighbor_str)
                                
                                # Cập nhật điểm cao nhất nếu một part xuất hiện nhiều lần trong window
                                current_best = frame_part_scores[frame_key].get(part_index, -1000.0)
                                if float(dist) > current_best:
                                    frame_part_scores[frame_key][part_index] = float(dist)
                                
                                if frame_key not in frame_info_cache:
                                    meta = video_meta[neighbor_n]
                                    pts_time = meta.get('pts_time', 0) if meta else 0
                                    
                                    # Tạo lại web_path cho frame hàng xóm
                                    prefix = video_id_to_path_prefix.get(video_id)
                                    if prefix:
                                        neighbor_web_path = f"Keyframes/{prefix}/{neighbor_str}.jpg"
                                    else:
                                        # Fallback an toàn (thay thế tên file)
                                        neighbor_web_path = web_path.replace(f"{frame_n_str}.jpg", f"{neighbor_str}.jpg")
                                        
                                    frame_info_cache[frame_key] = {
                                        "path": neighbor_web_path,
                                        "videoId": video_id,
                                        "pts_time": float(pts_time)
                                    }

        # 4. Lọc kết quả dựa trên số "common points"
        all_results = []
        for frame_key, parts_dict in frame_part_scores.items():
            common_count = len(parts_dict)
            if common_count >= min_common:
                # Frame này thỏa mãn điều kiện
                info = frame_info_cache[frame_key].copy()
                # Điểm tổng là tổng điểm các lần match tốt nhất vùng lân cận
                info['sum_score'] = sum(parts_dict.values())
                info['score'] = info['sum_score'] 
                info['common_count'] = common_count 
                all_results.append(info)
                summary_counter[info['videoId']] += 1
        
        print(f"[TRAKE.02] Tìm thấy {len(all_results)} frame chung (>= {min_common} parts)")
                
        # 5. Sắp xếp và định dạng output
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
                # Sắp xếp theo common_count (score) giảm dần, sau đó là thời gian
                sorted_items = sorted(items, key=lambda x: (-x['score'], x['pts_time']))
                final_grouped_results[video_id] = sorted_items[:top_k_final] # Cắt theo top_k final
            
            return jsonify({"results": final_grouped_results, "summary": sorted_summary})
        else:
            # Sắp xếp theo common_count (score) giảm dần, sau đó là thời gian
            final_results = sorted(all_results, key=lambda x: (-x['score'], x['pts_time']))
            return jsonify({"results": final_results[:top_k_final], "summary": sorted_summary}) # Cắt theo top_k final

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
# (CẬP NHẬT) API /search_asr
@app.route('/search_asr', methods=['POST'])
def search_asr():
    try:
        # (SỬA LỖI) Lấy JSON trực tiếp
        data = request.get_json()
        if data is None:
             return jsonify({"error": "Request phải là JSON"}), 400
             
        query_text = data['query'].lower() 
        top_k = int(data.get('top_k', 50))
        # (SỬA LỖI) Xử lý 'group' (là boolean true/false)
        group_results = data.get('group', False)
        
        if not bm25_asr_index:
            return jsonify({"error": "Index BM25 (ASR) chưa được khởi tạo."}), 500
        if not query_text:
            return jsonify({"results": [], "summary": {}})

        tokenized_query = query_text.split()
        
        scores = bm25_asr_index.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:top_k * 5 if group_results else top_k]

        top_k_documents = []
        for i in top_k_indices:
            score = scores[i]
            if score <= 0:
                continue
            doc = asr_data[i].copy() # Quan trọng: copy
            doc['score'] = float(score)
            top_k_documents.append(doc) 
            
        final_results = []
        processed_segments_key = set() 
        summary = {}
        
        for doc in top_k_documents: 
            video_id = doc['video_id']
            segment_index = doc['video_segment_index']
            doc_key = f"{video_id}_{segment_index}"
            if doc_key in processed_segments_key: continue
            
            # (Logic nối (stitching) giữ nguyên)
            def contains_keyword(text, query_tokens):
                text_lower = text.lower()
                for token in query_tokens:
                    if token in text_lower: return True
                return False
            
            current_merge_group = [doc]
            processed_segments_key.add(doc_key)
            video_segments_list = asr_video_map.get(video_id, [])

            prev_index = segment_index - 1
            if prev_index >= 0:
                prev_segment = video_segments_list[prev_index]
                if contains_keyword(prev_segment['text'], tokenized_query):
                    current_merge_group.insert(0, prev_segment)
                    processed_segments_key.add(f"{video_id}_{prev_index}")
                    
            next_index = segment_index + 1
            if next_index < len(video_segments_list):
                next_segment = video_segments_list[next_index]
                if contains_keyword(next_segment['text'], tokenized_query):
                    current_merge_group.append(next_segment)
                    processed_segments_key.add(f"{video_id}_{next_index}")

            if len(current_merge_group) == 1: 
                final_doc = doc
            else:
                final_doc = {
                    "video_id": video_id,
                    "text": " + ".join([seg['text'] for seg in current_merge_group]),
                    "start": current_merge_group[0]['start'],
                    "end": current_merge_group[-1]['end'],
                    "score": doc['score'] # Lấy điểm của segment gốc
                }

            media_info = media_info_cache.get(video_id, {})
            watch_url = media_info.get('watch_url')
            final_doc['watch_url'] = watch_url 
            
            target_start_time = final_doc['start'] 
            closest_frame_data = find_closest_keyframe(video_id, target_start_time)
            final_doc['frame_n'] = closest_frame_data.get('frame_n')
            final_doc['frame_idx'] = closest_frame_data.get('frame_idx')
            
            # (CẬP NHẬT) Thêm web_path cho ASR
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

