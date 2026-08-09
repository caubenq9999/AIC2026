import torch
from transformers import BlipProcessor, BlipForImageTextRetrieval
from PIL import Image
import time
import os
from concurrent.futures import ThreadPoolExecutor

class BlipReranker:
    def __init__(self, model_name="Salesforce/blip-itm-base-coco", device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"[{time.strftime('%H:%M:%S')}] Đang tải mô hình Reranker ({model_name}) lên {device}...")
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode
        print(f"[{time.strftime('%H:%M:%S')}] Tải mô hình Reranker hoàn tất.")

    def rerank(self, query, image_results, image_base_dir="."):
        """
        Re-ranks a list of image results based on their relevance to the query.
        
        Args:
            query (str): The text query.
            image_results (list): A list of dictionaries representing the initial top-K results.
                                  Each dictionary must have an 'image_path' key.
                                  Example: [{'image_path': 'Keyframes_L21/L21_V001/001.jpg', 'score': 0.9}, ...]
            image_base_dir (str): The base directory where the images are stored (usually the project root).
        
        Returns:
            list: The reranked list of results.
        """
        if not image_results:
            return []

        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Bắt đầu rerank {len(image_results)} ảnh cho query: '{query}'")

        reranked_results = []

        # 1. Load images cục bộ, song song hóa bằng thread pool (I/O-bound, PIL nhả GIL lúc decode JPEG).
        # (BỎ) fallback tải qua Localtunnel (evil-donkeys-walk.loca.lt) - đây là URL tạm thời của một
        # phiên dev cũ, gần như chắc chắn đã chết. Mỗi ảnh load local thất bại từng phải chờ tới 5s
        # timeout mạng cho fallback này -> đây là nguyên nhân lag nhiều khả năng nhất khi rerank nhiều ảnh.
        load_start = time.time()

        def load_one(item):
            full_path = os.path.join(image_base_dir, item['image_path'])
            try:
                return item, Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"[Reranker] Lỗi khi tải ảnh {full_path}: {e}")
                return item, None

        with ThreadPoolExecutor(max_workers=min(16, len(image_results))) as pool:
            loaded = list(pool.map(load_one, image_results))

        valid_results = []
        images = []
        for item, img in loaded:
            if img is None:
                item['itm_score'] = -999.0
                reranked_results.append(item)
            else:
                valid_results.append(item)
                images.append(img)

        load_time = time.time() - load_start

        if not images:
            print("Không tải được ảnh nào để rerank.")
            return image_results

        # 2. Process inputs in batches to avoid OOM
        infer_start = time.time()
        batch_size = 32 # Adjust based on available VRAM
        itm_scores = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_texts = [query] * len(batch_images)

                inputs = self.processor(images=batch_images, text=batch_texts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Compute matching scores
                outputs = self.model(**inputs)

                # The model outputs ITM logits (shape: [batch_size, 2])
                # We take the softmax and get the probability of class 1 (matching)
                probs = torch.nn.functional.softmax(outputs.itm_score, dim=1)
                match_probs = probs[:, 1].cpu().tolist()
                itm_scores.extend(match_probs)

        infer_time = time.time() - infer_start

        # 3. Assign scores and sort
        for item, score in zip(valid_results, itm_scores):
            item['itm_score'] = score
            reranked_results.append(item)

        # Sort by itm_score in descending order
        reranked_results.sort(key=lambda x: x.get('itm_score', -999.0), reverse=True)

        end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Reranking hoàn tất trong {end_time - start_time:.2f}s "
              f"(load ảnh: {load_time:.2f}s, inference: {infer_time:.2f}s).")

        return reranked_results

if __name__ == "__main__":
    # Test nhanh (Standalone test)
    print("Test độc lập BlipReranker")
    reranker = BlipReranker()
    # Tạo dummy data
    # dummy_results = [{'image_path': '...', 'score': 0}]
    # reranked = reranker.rerank("a cat", dummy_results)
    # print(reranked)
