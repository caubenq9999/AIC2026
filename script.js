// === (THÊM MỚI) BIẾN TOÀN CỤC CHO YOUTUBE PLAYER ===

let ytPlayer; // Biến giữ đối tượng player
let currentKeyframeMap = { fps: null }; // (THAY ĐỔI) Bỏ a, b, thêm fps
let videoTimeInterval; // Biến giữ interval để check thời gian
let currentLoadedYoutubeId = null; // (ĐỔI TÊN) Giữ ID YouTube đang tải
let currentLoadedInternalMapId = null; // (THÊM MỚI) Giữ ID map nội bộ đang tải
// === (THÊM MỚI) BIẾN TOÀN CỤC CHO SUBMISSION PANEL ===
let currentSubmissionVideoId = null; // ID video nội bộ (L21_V001)
let currentSubmissionMode = 'QA'; // Chế độ submit mặc định
let kisStartTime = null;
let trakeFrames = [];
// (FIX) Biến global cho searchMode để dùng trong display functions
let currentSearchMode = ''; // (THÊM MỚI) Global để tránh lỗi "not defined"
// ===== LAZY LOADING OBSERVER =====
// (SỬA LỖI) opacity giờ do CSS (.gallery-item.lazy) điều khiển hoàn toàn qua transition,
// không set inline style nữa -> tránh xung đột & mượt hơn khi cuộn nhanh.
// rootMargin nới ra 400px để ảnh kịp tải trước khi lọt vào khung nhìn (giảm hiện tượng "pop-in").
const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;

            img.onload = () => { img.classList.remove('lazy'); };
            img.onerror = () => { img.classList.remove('lazy'); img.alt = 'Lỗi tải ảnh'; };

            observer.unobserve(img);
        }
    });
}, {
    rootMargin: '400px',
    threshold: 0.01
});

// (SỬA LỖI) Placeholder cũ trỏ ra 1 URL Facebook CDN ngoài (giòn, tốn 1 request mạng cho MỌI
// thumbnail chưa tải). Thay bằng SVG nội tuyến (data URI) -> không tốn request, hiện tức thì.
const LAZY_PLACEHOLDER_SRC =
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='9'%3E%3Crect width='16' height='9' fill='%23eef1f4'/%3E%3C/svg%3E";

// ===== (THÊM MỚI) HÀM TẠO ẢNH VỚI LAZY LOADING =====
function createLazyImage(item) {
    const img = document.createElement('img');

    img.dataset.src = item.path;
    img.dataset.videoId = item.videoId;
    img.className = 'gallery-item lazy';
    img.loading = 'lazy'; // gợi ý thêm cho trình duyệt, bổ trợ cho IntersectionObserver
    img.src = LAZY_PLACEHOLDER_SRC;

    if (item.common_count !== undefined) {
        const label = currentSearchMode === 'trake-image' ? 'Common Images' : 'Common Parts';
        img.title = `${label}: ${item.common_count}\nTime: ${item.pts_time.toFixed(2)}s\nScore: ${item.score.toFixed(4)}`;
    } else if (item.matched_by !== undefined) {
        img.title = `Matched by: ${item.matched_by.join('+')}\nTime: ${item.pts_time.toFixed(2)}s\nScore: ${item.score.toFixed(4)}`;
    } else {
        img.title = `Time: ${item.pts_time.toFixed(2)}s\nScore: ${item.score.toFixed(4)}`;
    }

    img.addEventListener('click', () => { window.showImageDetail(item.path, img); });

    imageObserver.observe(img);

    return img;
}

// === (THÊM MỚI) HÀM NÀY SẼ ĐƯỢC YOUTUBE API TỰ ĐỘNG GỌI ===
function onYouTubeIframeAPIReady() {
    // Để trống, chúng ta sẽ tạo player khi cần
    console.log("YouTube IFrame API đã sẵn sàng.");
}
// === (THÊM MỚI) HÀM HELPER TÌM KIẾM NHỊ PHÂN (GIỐNG PYTHON) ===
function bisect_left(arr, x) {
    let low = 0, high = arr.length;
    while (low < high) {
        let mid = Math.floor((low + high) / 2);
        if (arr[mid] < x) low = mid + 1;
        else high = mid;
    }
    return low;
}
document.addEventListener('DOMContentLoaded', () => {
    // --- Lấy các đối tượng DOM ---
    const queryInput = document.getElementById('query-input');
    const searchButton = document.getElementById('search-button');
    const topKInput = document.getElementById('top-k-input');
    const windowSizeInput = document.getElementById('window-size-input'); // (THÊM MỚI) Nới khung
    const translateCheckbox = document.getElementById('translate-checkbox');
    const groupResultsCheckbox = document.getElementById('group-results-checkbox'); // (THÊM MỚI)
    const autoCropCheckbox = document.getElementById('auto-crop-checkbox'); // (THÊM MỚI) YOLOv8n auto-crop
    // (CẬP NHẬT) Query Expansion: nút "Mở rộng" + chip lựa chọn (bấm chọn 1 cái để search)
    const queryExpansionBox = document.getElementById('query-expansion-box');
    const expandQueryButton = document.getElementById('expand-query-button');
    const queryExpansionOptions = document.getElementById('query-expansion-options');
    const beit3WeightSlider = document.getElementById('beit3-weight-slider'); // (THÊM MỚI) Fusion CLIP/BEIT3
    const fusionWeightClipLabel = document.getElementById('fusion-weight-clip');
    const fusionWeightBeit3Label = document.getElementById('fusion-weight-beit3');
    // (THÊM MỚI) DOM cho danh sách ô nhập sự kiện của TRAKE
    const trakeInputArea = document.getElementById('trake-input-area');
    const trakeEventRows = document.getElementById('trake-event-rows');
    const trakeAddEventButton = document.getElementById('trake-add-event-button');
    const trakeMaxGapInput = document.getElementById('trake-max-gap-input');
    // (THÊM MỚI) DOM cho 3 ô riêng + tỷ lệ của Fusion Search (CLIP/OCR/ASR)
    const fusionInputArea = document.getElementById('fusion-input-area');
    const fusionQueryClip = document.getElementById('fusion-query-clip');
    const fusionQueryOcr = document.getElementById('fusion-query-ocr');
    const fusionQueryAsr = document.getElementById('fusion-query-asr');
    const fusionSearchWeightClip = document.getElementById('fusionsearch-weight-clip');
    const fusionSearchWeightOcr = document.getElementById('fusionsearch-weight-ocr');
    const fusionSearchWeightAsr = document.getElementById('fusionsearch-weight-asr');
    const fusionSearchWeightClipPct = document.getElementById('fusionsearch-weight-clip-pct');
    const fusionSearchWeightOcrPct = document.getElementById('fusionsearch-weight-ocr-pct');
    const fusionSearchWeightAsrPct = document.getElementById('fusionsearch-weight-asr-pct');
    const fusionSyncCheckbox = document.getElementById('fusion-sync-checkbox');
    const statusMessage = document.getElementById('status-message');
    const summaryBox = document.getElementById("summary-box");
    // (THÊM MỚI) DOM cho khu vực tải ảnh (single)
    const imageUploadArea = document.getElementById('image-upload-area');
    const imageUploadInput = document.getElementById('image-upload-input');
    const imagePreview = document.getElementById('image-preview');
    // (THÊM MỚI CHO TRAKE IMAGE) DOM cho multiple upload
    const multiImageUploadArea = document.getElementById('multi-image-upload-area');
    const multiImageUploadInput = document.getElementById('multi-image-upload-input');
    const multiImagePreview = document.getElementById('multi-image-preview');
    // DOM cho các container kết quả
    const imageContainer = document.getElementById('image-container');
    const asrContainer = document.getElementById('asr-container');
    // DOM cho panel detail
    const detailBox = document.getElementById('image-detail');
    const closeDetailButton = document.getElementById('close-detail');
    const noVideoLinkSpan = document.getElementById('no-video-link');
    // DOM cho Video Player
    const videoPlayerArea = document.getElementById('video-player-area');
    const videoPlayerTitle = document.getElementById('video-player-title');
    const closeVideoPlayerButton = document.getElementById('close-video-player');

    // DOM cho Panel thời gian thực
    const realTimePanel = document.getElementById('real-time-panel');
    const currentVideoTimeSpan = document.getElementById('current-video-time');
    const currentFrameIndexSpan = document.getElementById('current-frame-index');
    // DOM cho các nút chọn chế độ tìm kiếm
    const modeSemanticRadio = document.getElementById('mode-semantic');
    const modeOcrRadio = document.getElementById('mode-ocr');
    const modeAsrRadio = document.getElementById('mode-asr');
    const modeFusionRadio = document.getElementById('mode-fusion'); // (THÊM MỚI) Gộp CLIP+OCR+ASR
    const modeSimilarRadio = document.getElementById('mode-similar'); // (THÊM MỚI)
    const modeTrake02Radio = document.getElementById('mode-trake-02'); // (THÊM MỚI CHO TRAKE.02)
    const modeTrakeImageRadio = document.getElementById('mode-trake-image'); // (THÊM MỚI CHO TRAKE IMAGE)
    const translateBox = document.querySelector('.translate-box');
    const groupResultsBox = document.querySelector('.group-box'); // (THÊM MỚI)
    const windowSizeBox = document.querySelector('.window-size-box'); // (THÊM MỚI) Nới khung
    // === (THÊM MỚI) DOM CHO SUBMISSION PANEL ===
    const submissionPanel = document.getElementById('submission-panel');
    const submissionLog = document.getElementById('submission-log');
    const submitAnswerButton = document.getElementById('submit-answer-button');

    // (SỬA LỖI) Nút X của Submission Panel (thêm vào)
    const closeSubmissionPanelButton = document.getElementById('close-submission-panel');

    // Các nút radio chọn chế độ submit
    const submitModeRadios = document.querySelectorAll('input[name="submit-mode"]');

    // UI cho chế độ QA
    const qaModeUI = document.getElementById('qa-mode-ui');
    const qaAnswerInput = document.getElementById('qa-answer-input');
    // UI cho chế độ KIS
    const kisModeUI = document.getElementById('kis-mode-ui');
    const kisClickButton = document.getElementById('kis-click-button');
    const kisResetButton = document.getElementById('kis-reset-button');
    const kisStartTimeSpan = document.getElementById('kis-start-time');
    const kisEndTimeSpan = document.getElementById('kis-end-time');
    const kisManualStartInput = document.getElementById('kis-manual-start-input');
    const kisManualEndInput = document.getElementById('kis-manual-end-input');
    const kisManualSetButton = document.getElementById('kis-manual-set-button');
    // UI cho chế độ TRAKE
    const trakeModeUI = document.getElementById('trake-mode-ui');
    const trakeClickButton = document.getElementById('trake-click-button');
    const trakeUndoButton = document.getElementById('trake-undo-button');
    const trakeFramesListSpan = document.getElementById('trake-frames-list');
    const trakeManualInput = document.getElementById('trake-manual-frame-input');
    const trakeManualAddButton = document.getElementById('trake-manual-add-button');
    // === KẾT THÚC DOM MỚI ===
    let currentNeighborPaths = [];
    let currentDetailPath = "";
    // --- GẮN CÁC SỰ KIỆN ---
    videoPlayerTitle.addEventListener('click', () => {
        videoPlayerArea.classList.toggle('full-screen');
    });
    searchButton.addEventListener('click', performSearch);
    queryInput.addEventListener('keydown', (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            performSearch();
        }
    });
    // (THÊM MỚI) Sự kiện khi chọn tệp ảnh (single)
    imageUploadInput.addEventListener('change', () => {
        const file = imageUploadInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    });
    // (THÊM MỚI CHO TRAKE IMAGE) Sự kiện khi chọn multiple ảnh
    multiImageUploadInput.addEventListener('change', () => {
        const files = multiImageUploadInput.files;
        if (files.length < 2) {
            multiImagePreview.innerHTML = '<p style="color: red;">Cần ít nhất 2 ảnh!</p>';
            multiImagePreview.classList.add('hidden');
            return;
        }
        multiImagePreview.innerHTML = '';
        multiImagePreview.classList.remove('hidden');
        Array.from(files).forEach(file => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                multiImagePreview.appendChild(img);
            };
            reader.readAsDataURL(file);
        });
    });
    // (CẬP NHẬT) Đóng panel chi tiết ảnh -> Tắt CẢ BA panel
    closeDetailButton.addEventListener("click", () => {
        // [SỬA LỖI !important] Thay vì .remove("visible"), dùng .add("hidden")
        detailBox.classList.add("hidden");
        submissionPanel.classList.add("hidden"); // (THÊM MỚI) Ẩn submit

        // Gọi hàm đóng video (vì nó cũng tắt submission)
        closeVideoPlayerButton.click(); // Gọi hàm đóng video

        document.querySelectorAll(".gallery-item.selected").forEach(el => el.classList.remove("selected"));
        // (THÊM MỚI) Dọn dẹp tất cả các ID
        currentLoadedYoutubeId = null;
        currentLoadedInternalMapId = null;
        currentSubmissionVideoId = null;
    });
    // (CẬP NHẬT) Đóng video player -> Chỉ tắt video
    closeVideoPlayerButton.addEventListener("click", () => {
        // [SỬA LỖI !important]
        videoPlayerArea.classList.add("hidden");

        if (ytPlayer && typeof ytPlayer.stopVideo === 'function') { // Thêm check
            ytPlayer.stopVideo(); // Dùng API để dừng
        }
        if (videoTimeInterval) {
            clearInterval(videoTimeInterval); // Dừng interval
        }
    });
    // (THÊM MỚI) Nút đóng của Submission Panel
    closeSubmissionPanelButton.addEventListener("click", () => {
        // [SỬA LỖI !important]
        submissionPanel.classList.add("hidden");
    });
    // (THÊM MỚI) Cập nhật nhãn % khi kéo thanh trượt fusion CLIP/BEIT3
    if (beit3WeightSlider) {
        beit3WeightSlider.addEventListener('input', () => {
            const clipPct = parseInt(beit3WeightSlider.value, 10);
            fusionWeightClipLabel.textContent = `${clipPct}%`;
            fusionWeightBeit3Label.textContent = `${100 - clipPct}%`;
        });
    }

    // (THÊM MỚI) Fusion Search: đồng bộ nội dung 3 ô CLIP/OCR/ASR khi tích "Đồng bộ"
    function syncFusionQueryBoxes(sourceInput) {
        if (!fusionSyncCheckbox.checked) return;
        [fusionQueryClip, fusionQueryOcr, fusionQueryAsr].forEach(input => {
            if (input !== sourceInput) input.value = sourceInput.value;
        });
    }
    fusionQueryClip.addEventListener('input', () => syncFusionQueryBoxes(fusionQueryClip));
    fusionQueryOcr.addEventListener('input', () => syncFusionQueryBoxes(fusionQueryOcr));
    fusionQueryAsr.addEventListener('input', () => syncFusionQueryBoxes(fusionQueryAsr));
    // Bật lại "Đồng bộ" -> đồng bộ ngay theo nội dung ô CLIP để tránh 3 ô đang lệch nhau
    fusionSyncCheckbox.addEventListener('change', () => {
        if (fusionSyncCheckbox.checked) syncFusionQueryBoxes(fusionQueryClip);
    });

    // (THÊM MỚI) Fusion Search: cập nhật nhãn % tỷ lệ CLIP/OCR/ASR (chuẩn hoá theo tổng 3 thanh trượt)
    function updateFusionSearchWeightLabels() {
        const wClip = parseInt(fusionSearchWeightClip.value, 10);
        const wOcr = parseInt(fusionSearchWeightOcr.value, 10);
        const wAsr = parseInt(fusionSearchWeightAsr.value, 10);
        const total = wClip + wOcr + wAsr;
        const pct = (w) => total > 0 ? Math.round((w / total) * 100) : 0;
        fusionSearchWeightClipPct.textContent = `${pct(wClip)}%`;
        fusionSearchWeightOcrPct.textContent = `${pct(wOcr)}%`;
        fusionSearchWeightAsrPct.textContent = `${pct(wAsr)}%`;
    }
    [fusionSearchWeightClip, fusionSearchWeightOcr, fusionSearchWeightAsr].forEach(slider => {
        slider.addEventListener('input', updateFusionSearchWeightLabels);
    });

    // (THÊM MỚI) TRAKE: quản lý danh sách ô nhập sự kiện (thêm / xóa / đánh số lại)
    const TRAKE_MIN_EVENTS = 2; // dưới 2 sự kiện thì không còn là "chuỗi"
    const TRAKE_PLACEHOLDER_EXAMPLES = ['chạy đà', 'giậm nhảy', 'bay qua xà', 'tiếp đất'];

    function addTrakeEventRow(value = '') {
        const row = document.createElement('div');
        row.className = 'trake-event-row';

        const indexBadge = document.createElement('span');
        indexBadge.className = 'trake-event-index';
        row.appendChild(indexBadge);

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'trake-event-input';
        input.value = value;
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                performSearch();
            }
        });
        row.appendChild(input);

        const removeButton = document.createElement('button');
        removeButton.type = 'button';
        removeButton.className = 'trake-event-remove';
        removeButton.textContent = '✕';
        removeButton.title = 'Xóa sự kiện này';
        removeButton.addEventListener('click', () => {
            row.remove();
            refreshTrakeEventRows();
        });
        row.appendChild(removeButton);

        trakeEventRows.appendChild(row);
        refreshTrakeEventRows();
        return input;
    }

    // Đánh số lại thứ tự + cập nhật placeholder + khoá nút xóa khi đã ở mức tối thiểu
    function refreshTrakeEventRows() {
        const rows = trakeEventRows.querySelectorAll('.trake-event-row');
        rows.forEach((row, i) => {
            row.querySelector('.trake-event-index').textContent = i + 1;
            const example = TRAKE_PLACEHOLDER_EXAMPLES[i];
            row.querySelector('.trake-event-input').placeholder =
                `Sự kiện ${i + 1}` + (example ? ` (ví dụ: ${example})` : '');
            row.querySelector('.trake-event-remove').disabled = rows.length <= TRAKE_MIN_EVENTS;
        });
    }

    // Bỏ qua ô để trống -> người dùng không cần xóa ô thừa mới search được
    function getTrakeEventQueries() {
        return Array.from(trakeEventRows.querySelectorAll('.trake-event-input'))
            .map(input => input.value.trim())
            .filter(v => v);
    }

    trakeAddEventButton.addEventListener('click', () => {
        addTrakeEventRow().focus();
    });

    // Khởi tạo sẵn 4 ô theo đúng ví dụ chuỗi sự kiện của BTC (nhảy cao)
    for (let i = 0; i < 4; i++) addTrakeEventRow();

    // (CẬP NHẬT) Query Expansion: bấm nút -> gọi /expand_query -> hiện 3 lựa chọn -> bấm chọn 1 cái để search luôn
    expandQueryButton.addEventListener('click', async () => {
        const q = queryInput.value.trim();
        if (!q) {
            statusMessage.textContent = 'Nhập câu truy vấn trước khi mở rộng.';
            return;
        }
        const originalText = expandQueryButton.textContent;
        expandQueryButton.disabled = true;
        expandQueryButton.textContent = 'Đang mở rộng...';
        queryExpansionOptions.innerHTML = '';
        queryExpansionOptions.classList.add('hidden');
        try {
            const response = await fetch('http://localhost:5000/expand_query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: q })
            });
            if (!response.ok) throw new Error(`Lỗi server: ${response.statusText}`);
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            const variants = data.variants || [];
            if (variants.length === 0) {
                statusMessage.textContent = 'Không mở rộng được câu truy vấn (Groq không trả kết quả hợp lệ).';
                return;
            }
            variants.forEach(variant => {
                const option = document.createElement('button');
                option.type = 'button';
                option.className = 'query-expansion-option';
                option.textContent = variant; // (an toàn) textContent, tránh innerHTML với text do LLM sinh ra
                option.addEventListener('click', () => {
                    queryInput.value = variant;
                    queryExpansionOptions.innerHTML = '';
                    queryExpansionOptions.classList.add('hidden');
                    performSearch(); // Chọn xong search luôn cho nhanh
                });
                queryExpansionOptions.appendChild(option);
            });
            queryExpansionOptions.classList.remove('hidden');
        } catch (error) {
            console.error('Lỗi khi mở rộng câu truy vấn:', error);
            statusMessage.textContent = `Lỗi mở rộng câu truy vấn: ${error.message}`;
        } finally {
            expandQueryButton.disabled = false;
            expandQueryButton.textContent = originalText;
        }
    });

    // Sự kiện khi thay đổi chế độ tìm kiếm
    modeSemanticRadio.addEventListener('change', updateControls);
    modeOcrRadio.addEventListener('change', updateControls);
    modeAsrRadio.addEventListener('change', updateControls);
    modeFusionRadio.addEventListener('change', updateControls); // (THÊM MỚI) Gộp CLIP+OCR+ASR
    modeSimilarRadio.addEventListener('change', updateControls); // (THÊM MỚI)
    modeTrake02Radio.addEventListener('change', updateControls); // (THÊM MỚI CHO TRAKE.02)
    modeTrakeImageRadio.addEventListener('change', updateControls); // (THÊM MỚI CHO TRAKE IMAGE)
    updateControls(); // Chạy lần đầu
    document.addEventListener('keydown', (event) => {
        if (event.key === "Escape") {
            // [SỬA LỖI !important] Check bằng .hidden
            if (!detailBox.classList.contains("hidden")) {
                // Giả lập click nút đóng chi tiết (vì nó đóng cả 3)
                closeDetailButton.click();
            }
        }
        // [SỬA LỖI !important] Check bằng .hidden
        if (detailBox.classList.contains("hidden")) return;
        if (event.key === 'ArrowLeft') navigateNeighbor(-1);
        if (event.key === 'ArrowRight') navigateNeighbor(1);
    });
    // === (THÊM MỚI) CÁC SỰ KIỆN CHO SUBMISSION PANEL ===

    // 1. Thay đổi chế độ (QA, KIS, TRAKE)
    submitModeRadios.forEach(radio => {
        radio.addEventListener('change', updateSubmissionUIVisibility);
    });
    // 2. Nút Click của KIS
    kisClickButton.addEventListener('click', () => {
        const currentTime = parseFloat(currentVideoTimeSpan.textContent);
        if (isNaN(currentTime)) {
            alert("Chưa có thời gian video!");
            return;
        }
        if (ytPlayer && typeof ytPlayer.pauseVideo === 'function') {
            ytPlayer.pauseVideo(); // Tự động pause video khi click
        }
        if (kisStartTime === null) {
            kisStartTime = currentTime;
            kisStartTimeSpan.textContent = currentTime.toFixed(2);
            kisEndTimeSpan.textContent = "N/A";
            kisClickButton.textContent = "Click (Set End)";
        } else {
            kisEndTimeSpan.textContent = currentTime.toFixed(2);
            kisClickButton.textContent = "Click (Set Start)"; // Quay lại trạng thái chờ
        }
        kisManualStartInput.value = "";
        kisManualEndInput.value = "";
    });
    // 2b. Nút Set (Manual) của KIS
    kisManualSetButton.addEventListener('click', () => {
        const startText = kisManualStartInput.value.trim();
        const endText = kisManualEndInput.value.trim();

        if (!startText || !endText) {
            alert("Vui lòng nhập cả Start Time và End Time.");
            return;
        }

        const startTime = parseFloat(startText);
        const endTime = parseFloat(endText);

        if (isNaN(startTime) || isNaN(endTime) || startTime < 0 || endTime < 0) {
            alert("Thời gian Start hoặc End không hợp lệ.");
            return;
        }

        if (endTime <= startTime) {
            alert("End Time phải lớn hơn Start Time.");
            return;
        }

        // Cập nhật biến global và UI
        kisStartTime = startTime; // Cập nhật biến state
        kisStartTimeSpan.textContent = startTime.toFixed(2);
        kisEndTimeSpan.textContent = endTime.toFixed(2);

        // Reset lại nút click (nếu nó đang ở trạng thái "Set End")
        kisClickButton.textContent = "Click (Set Start)";
    });
    // 3. Nút Reset của KIS
    kisResetButton.addEventListener('click', () => {
        kisStartTime = null;
        kisStartTimeSpan.textContent = "N/A";
        kisEndTimeSpan.textContent = "N/A";
        kisClickButton.textContent = "Click (Set Start)";
    });
    // 4. Nút Click của TRAKE
    trakeClickButton.addEventListener('click', () => {
        const currentFrame = currentFrameIndexSpan.textContent;
        if (currentFrame === "N/A" || currentFrame === "Đang tải..." || currentFrame === "Lỗi") {
            alert("Chưa có frame index hợp lệ!");
            return;
        }
        if (ytPlayer && typeof ytPlayer.pauseVideo === 'function') {
            ytPlayer.pauseVideo(); // Tự động pause video khi click
        }

        const frameId = parseInt(currentFrame);
        if (!trakeFrames.includes(frameId)) { // Chỉ thêm nếu chưa có
            trakeFrames.push(frameId);
            trakeFramesListSpan.textContent = JSON.stringify(trakeFrames);
        }
    });
    // (THÊM MỚI) 4b. Nút Add (Manual) của TRAKE
    trakeManualAddButton.addEventListener('click', () => {
        const frameIdText = trakeManualInput.value.trim();
        if (!frameIdText) {
            alert("Vui lòng nhập một Frame Index.");
            return;
        }

        const frameId = parseInt(frameIdText);
        if (isNaN(frameId) || frameId < 0) {
            alert("Frame Index không hợp lệ.");
            return;
        }

        if (!trakeFrames.includes(frameId)) { // Chỉ thêm nếu chưa có
            trakeFrames.push(frameId);
            // (THÊM MỚI) Sắp xếp lại mảng theo thứ tự thời gian (số)
            trakeFrames.sort((a, b) => a - b);
            trakeFramesListSpan.textContent = JSON.stringify(trakeFrames);
        }

        trakeManualInput.value = ""; // Xóa input sau khi thêm
    });
    // 5. Nút Undo của TRAKE
    trakeUndoButton.addEventListener('click', () => {
        if (trakeFrames.length > 0) {
            trakeFrames.pop(); // Xóa phần tử cuối
            trakeFramesListSpan.textContent = JSON.stringify(trakeFrames);
        }
    });
    // 6. Nút SUBMIT
    submitAnswerButton.addEventListener('click', submitAnswer);
    // (THÊM MỚI) DOM cho 2 ô input ID
    const btcEvalIdInput = document.getElementById('btc-evaluation-id');
    const btcSessIdInput = document.getElementById('btc-session-id');
    // === KẾT THÚC SỰ KIỆN MỚI ===
    // --- CÁC HÀM CHÍNH ---
    // (THÊM MỚI) Reset các form trong submission panel
    function resetSubmissionForms() {

        // Reset QA
        qaAnswerInput.value = "";
        // Reset KIS
        kisResetButton.click(); // Giả lập click nút reset
        // Reset TRAKE
        trakeFrames = [];
        trakeFramesListSpan.textContent = "[]";
        // Reset log
        // submissionLog.textContent = ""; // Đã di chuyển vào hàm submitAnswer
        // Đặt lại chế độ mặc định (QA)
        document.getElementById('submit-mode-qa').checked = true;
        updateSubmissionUIVisibility();
    }
    // (THÊM MỚI) Ẩn/hiện UI của submission panel
    function updateSubmissionUIVisibility() {
        currentSubmissionMode = document.querySelector('input[name="submit-mode"]:checked').value;

        const badge = document.getElementById('submission-mode-badge'); // (THÊM MỚI)
        if (badge) badge.textContent = currentSubmissionMode;

        // (SỬA LỖI) Dùng classList thay vì style.display trực tiếp để nhất quán với .hidden
        qaModeUI.classList.add('hidden');
        kisModeUI.classList.add('hidden');
        trakeModeUI.classList.add('hidden');
        if (currentSubmissionMode === 'QA') {
            qaModeUI.classList.remove('hidden');
        } else if (currentSubmissionMode === 'KIS') {
            kisModeUI.classList.remove('hidden');
        } else if (currentSubmissionMode === 'TRAKE') {
            trakeModeUI.classList.remove('hidden');
        }
        // Cập nhật thông tin chung (thời gian, frame)
        // (Sửa lỗi QA: Cập nhật thông tin khi chuyển tab)
        const currentTime = parseFloat(currentVideoTimeSpan.textContent);
        const currentFrame = currentFrameIndexSpan.textContent;
        document.querySelectorAll('.submission-video-id').forEach(span => {
            span.textContent = currentSubmissionVideoId || "N/A";
        });
        document.querySelectorAll('.submission-time').forEach(span => {
            span.textContent = isNaN(currentTime) ? "N/A" : currentTime.toFixed(2);
        });
        document.querySelectorAll('.submission-frame-idx').forEach(span => {
            span.textContent = currentFrame;
        });
    }
    // (CẬP NHẬT) Hàm Submit Answer (Phiên bản cuối cùng, cố gắng tạo đúng cấu trúc)
    async function submitAnswer() {
        // Xóa log cũ ngay khi bắt đầu submit
        submissionLog.textContent = "";

        // 1. (THÊM MỚI) Đọc 2 ID từ UI
        const evalId = btcEvalIdInput.value.trim();
        const sessId = btcSessIdInput.value.trim();

        if (!evalId || !sessId) {
            const errMsg = "LỖI: Vui lòng nhập EVALUATION_ID và SESSION_ID (ở góc trên bên phải).";
            alert(errMsg);
            submissionLog.textContent = errMsg;
            return;
        }

        if (!currentSubmissionVideoId) {
            alert("Lỗi: Không có video ID nào được chọn.");
            return;
        }

        // 2. (GIỮ NGUYÊN) Tạo payload câu trả lời (QA/KIS/TRAKE)
        let answerPayload = {}; // Đây là JSON gốc cho BTC
        let isValid = false;

        if (currentSubmissionMode === 'QA') {
            const answer = qaAnswerInput.value.trim();
            const time_sec = parseFloat(currentVideoTimeSpan.textContent);
            if (!answer) { /* ... (phần check lỗi của bạn) ... */ return; }
            if (isNaN(time_sec)) { /* ... (phần check lỗi của bạn) ... */ return; }

            const time_ms = Math.round(time_sec * 1000);
            const answerString = `QA-${answer}-${currentSubmissionVideoId}-${time_ms}`;

            answerPayload = {
                "answerSets": [{ "answers": [{ "text": answerString }] }]
            };
            isValid = true;
        } else if (currentSubmissionMode === 'KIS') {
            const start_sec = parseFloat(kisStartTimeSpan.textContent);
            const end_sec = parseFloat(kisEndTimeSpan.textContent);
            if (isNaN(start_sec) || isNaN(end_sec)) { /* ... */ return; }
            if (end_sec < start_sec) { /* ... */ return; }

            const start_ms = Math.round(start_sec * 1000);
            const end_ms = Math.round(end_sec * 1000);

            answerPayload = {
                "answerSets": [{ "answers": [{ "mediaItemName": currentSubmissionVideoId, "start": start_ms, "end": end_ms }] }]
            };
            isValid = true;
        } else if (currentSubmissionMode === 'TRAKE') {
            if (trakeFrames.length === 0) { /* ... */ return; }
            trakeFrames.sort((a, b) => a - b);
            const framesString = trakeFrames.join(',');
            const textString = `TR-${currentSubmissionVideoId}-${framesString}`;

            answerPayload = {
                "answerSets": [{ "answers": [{ "text": textString }] }]
            };
            isValid = true;
        }

        if (!isValid) {
            alert("Chế độ submit không hợp lệ.");
            return;
        }

        // 3. (THAY ĐỔI) Tạo "Gói Hàng" (Wrapper) gửi cho app.py
        const wrapperPayload = {
            evaluation_id: evalId,
            session_id: sessId,
            answer_payload: answerPayload // Gói câu trả lời gốc vào bên trong
        };

        // 4. (THAY ĐỔI) Stringify và Gửi "Gói Hàng"
        const finalRequestString = JSON.stringify(wrapperPayload, null, 2);

        submissionLog.textContent = "Đang gửi...\n" + finalRequestString;

        try {
            const response = await fetch('http://localhost:5000/submit_answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: finalRequestString // Gửi Gói Hàng (wrapper)
            });

            const result = await response.json();

            if (response.ok && result.status === 'success') {
                // Hiển thị payload GỐC đã gửi cho BTC (lấy từ log của server)
                // Hoặc đơn giản là hiển thị lại payload wrapper
                submissionLog.textContent = "GỬI THÀNH CÔNG ĐẾN BTC:\n" + (result.btc_response || "OK");

                // Reset form sau khi thành công
                resetSubmissionForms();
            } else {
                // Lỗi từ server (Python) hoặc từ BTC
                const btcError = result.btc_response ? `\nBTC Response: ${result.btc_response}` : '';
                throw new Error(result.message + btcError);
            }
        } catch (error) {
            console.error('Lỗi khi submit:', error);
            submissionLog.textContent = "GỬI THẤT BẠI:\n" + error.message;
        }
    }
    function formatTime(totalSeconds) {
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60);
        return `${minutes} phút ${seconds} giây`;
    }
    // (CẬP NHẬT) Hàm updateControls
    function updateControls() {
        const searchMode = document.querySelector('input[name="search-mode"]:checked').value;
        currentSearchMode = searchMode; // (THÊM MỚI) Cập nhật biến global

        // (SỬA LỖI) Khi chuyển chế độ, TẮT TẤT CẢ
        // [SỬA LỖI !important]
        if (!detailBox.classList.contains("hidden")) {
            closeDetailButton.click(); // Giả lập đóng tắt cả
        }

        // (THÊM MỚI) Ẩn/hiện các tùy chọn
        translateBox.classList.add('hidden');
        if (queryExpansionBox) queryExpansionBox.classList.add('hidden'); // (THÊM MỚI)
        groupResultsBox.classList.add('hidden'); // Ẩn group box
        if (windowSizeBox) windowSizeBox.classList.add('hidden'); // Ẩn window size
        imageUploadArea.classList.add('hidden'); // Ẩn single upload mặc định
        multiImageUploadArea.classList.add('hidden'); // Ẩn multiple upload mặc định
        fusionInputArea.classList.add('hidden'); // (THÊM MỚI) Ẩn 3-ô Fusion mặc định
        trakeInputArea.classList.add('hidden'); // (THÊM MỚI) Ẩn danh sách ô sự kiện TRAKE mặc định

        // Logic Ẩn/Hiện ô query và upload areas
        if (searchMode === 'similar') {
            // Chế độ Similar: Ẩn text, Hiện single upload
            queryInput.classList.add('hidden');
            imageUploadArea.classList.remove('hidden');
            groupResultsBox.classList.remove('hidden');
        } else if (searchMode === 'trake-image') {
            // (THÊM MỚI CHO TRAKE IMAGE): Ẩn text, Hiện multiple upload
            queryInput.classList.add('hidden');
            multiImageUploadArea.classList.remove('hidden');
            groupResultsBox.classList.remove('hidden');
            if (windowSizeBox) windowSizeBox.classList.remove('hidden');
        } else if (searchMode === 'fusion') {
            // (THÊM MỚI) Chế độ Fusion: Ẩn ô text chung, hiện 3 ô riêng CLIP/OCR/ASR + tỷ lệ
            queryInput.classList.add('hidden');
            fusionInputArea.classList.remove('hidden');
            groupResultsBox.classList.remove('hidden');
            translateBox.classList.remove('hidden'); // Áp dụng cho riêng nhánh CLIP
        } else if (searchMode === 'trake-02') {
            // (CẬP NHẬT) TRAKE: mỗi sự kiện một ô riêng (thêm/xóa được) thay cho việc ngăn bằng ';'.
            // Không cần "Nới khung ±frame" (đã chuyển sang căn chỉnh theo thứ tự thời gian) và cũng
            // không cần "Nhóm theo video" vì mỗi video vốn đã là 1 chuỗi.
            queryInput.classList.add('hidden');
            trakeInputArea.classList.remove('hidden');
            translateBox.classList.remove('hidden'); // Dịch riêng từng sự kiện sang tiếng Anh cho CLIP
        } else {
            // Các chế độ text-based: Hiện text, Ẩn uploads
            queryInput.classList.remove('hidden');
            groupResultsBox.classList.remove('hidden');
            if (searchMode === 'semantic') {
                translateBox.classList.remove('hidden');
                // (THÊM MỚI) Query Expansion chỉ áp dụng cho Semantic Search
                if (queryExpansionBox) queryExpansionBox.classList.remove('hidden');
            }
        }

        // (SỬA LỖI) Logic ASR/Image container
        if (searchMode === 'asr') {
            imageContainer.classList.add('hidden');
            asrContainer.classList.remove('hidden');
        } else {
            // Mặc định (semantic, ocr, similar, trake-02, trake-image) là hiện image container
            imageContainer.classList.remove('hidden');
            asrContainer.classList.add('hidden');
        }
    }
    // (CẬP NHẬT) Hàm performSearch
    async function performSearch() {
        // (SỬA LỖI) Chặn double-submit: bấm nhanh 2 lần trước đây tạo 2 request chồng nhau,
        // kết quả hiển thị phụ thuộc request nào trả về sau (không đoán trước được).
        if (searchButton.disabled) return;
        searchButton.disabled = true;
        const originalButtonText = searchButton.textContent;
        searchButton.textContent = 'Đang tìm...';
        const resetSearchButton = () => {
            searchButton.disabled = false;
            searchButton.textContent = originalButtonText;
        };

        const query = queryInput.value.trim();

        statusMessage.textContent = 'Đang tìm kiếm...';
        imageContainer.innerHTML = ''; asrContainer.innerHTML = ''; summaryBox.innerHTML = '';
        queryExpansionOptions.classList.add('hidden'); queryExpansionOptions.innerHTML = '';

        // (SỬA LỖI) Đóng tất cả panel khi tìm kiếm
        // [SỬA LỖI !important]
        if (!detailBox.classList.contains("hidden")) {
            closeDetailButton.click();
        }

        const top_k = parseInt(topKInput.value, 10);
        const searchMode = document.querySelector('input[name="search-mode"]:checked').value;
        currentSearchMode = searchMode; // (FIX) Set global searchMode
        const group_results = groupResultsCheckbox.checked; // (THÊM MỚI)

        let endpoint = '';
        let fetchOptions = {}; // (THAY ĐỔI) Dùng object options
        if (searchMode === 'similar') {
            const imageFile = imageUploadInput.files[0];
            if (!imageFile) {
                statusMessage.textContent = 'Vui lòng tải lên một ảnh để tìm kiếm.';
                resetSearchButton();
                return;
            }
            endpoint = '/search_similar_image';

            // (QUAN TRỌNG) Dùng FormData cho tệp
            const formData = new FormData();
            formData.append('image_file', imageFile);
            formData.append('top_k', top_k);
            formData.append('group', group_results); // (THÊM MỚI)
            // (THÊM MỚI) Gửi auto_crop từ checkbox Toggle
            formData.append('auto_crop', autoCropCheckbox.checked ? 'true' : 'false');
            // (THÊM MỚI) Gửi trọng số fusion CLIP/BEIT3 từ thanh trượt (0.0 - 1.0)
            if (beit3WeightSlider) {
                formData.append('beit3_weight', (parseInt(beit3WeightSlider.value, 10) / 100).toString());
            }

            fetchOptions = {
                method: 'POST',
                body: formData
                // Không set Content-Type, trình duyệt tự làm
            };
        } else if (searchMode === 'trake-image') { // (THÊM MỚI CHO TRAKE IMAGE)
            const imageFiles = multiImageUploadInput.files;
            if (imageFiles.length < 2) {
                statusMessage.textContent = 'Cần ít nhất 2 ảnh để tìm giao.';
                resetSearchButton();
                return;
            }
            endpoint = '/search_trake_image';

            const formData = new FormData();
            Array.from(imageFiles).forEach(file => {
                formData.append('image_files', file);
            });
            formData.append('top_k', top_k);
            formData.append('group', group_results);
            if (windowSizeInput) formData.append('window_size', windowSizeInput.value);

            fetchOptions = {
                method: 'POST',
                body: formData
            };
        } else if (searchMode === 'fusion') {
            // (THÊM MỚI) Fusion: 3 ô riêng CLIP/OCR/ASR + tỷ lệ trọng số, gộp bằng RRF có trọng số
            const qClip = fusionQueryClip.value.trim();
            const qOcr = fusionQueryOcr.value.trim();
            const qAsr = fusionQueryAsr.value.trim();
            if (!qClip && !qOcr && !qAsr) {
                statusMessage.textContent = 'Vui lòng nhập ít nhất 1 trong 3 ô (CLIP/OCR/ASR).';
                resetSearchButton();
                return;
            }
            endpoint = '/search_fusion';
            const payload = {
                query_clip: qClip,
                query_ocr: qOcr,
                query_asr: qAsr,
                weight_clip: parseInt(fusionSearchWeightClip.value, 10),
                weight_ocr: parseInt(fusionSearchWeightOcr.value, 10),
                weight_asr: parseInt(fusionSearchWeightAsr.value, 10),
                top_k: top_k,
                group: group_results,
                translate: translateCheckbox.checked
            };
            fetchOptions = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            };
        } else if (searchMode === 'trake-02') {
            // (CẬP NHẬT) TRAKE: gom nội dung các ô sự kiện thành mảng, giữ nguyên thứ tự hiển thị
            const events = getTrakeEventQueries();
            if (events.length < 2) {
                statusMessage.textContent = 'TRAKE cần ít nhất 2 sự kiện — hãy điền thêm ô (bấm "+ Thêm sự kiện" nếu cần).';
                resetSearchButton();
                return;
            }
            endpoint = '/search_trake_02';
            const maxGap = parseFloat(trakeMaxGapInput.value);
            const payload = {
                events: events,
                top_k: top_k,
                translate: translateCheckbox.checked,
                // Vượt ngưỡng này giữa 2 sự kiện liên tiếp thì backend trừ điểm nặng
                max_gap_seconds: isNaN(maxGap) ? 30 : maxGap
            };
            fetchOptions = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            };
        } else {
            // Các chế độ dùng text
            if (!query) {
                statusMessage.textContent = 'Vui lòng nhập nội dung tìm kiếm.';
                resetSearchButton();
                return;
            }

            let payload = {
                query: query,
                top_k: top_k,
                group: group_results // (THÊM MỚI)
            };

            if (searchMode === 'semantic') {
                endpoint = '/search';
                payload.translate = translateCheckbox.checked;
            }
            else if (searchMode === 'ocr') {
                endpoint = '/search_ocr';
            }
            else if (searchMode === 'asr') {
                endpoint = '/search_asr';
            }

            fetchOptions = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            };
        }
        // (CẬP NHẬT) Fetch call dùng chung
        try {
            const response = await fetch('http://localhost:5000' + endpoint, fetchOptions);
            if (!response.ok) throw new Error(`Lỗi server: ${response.statusText}`);
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // (THÊM MỚI) TRAKE temporal trả về dạng CHUỖI SỰ KIỆN (mỗi video 1 chuỗi N frame),
            // khác hẳn dạng danh sách frame rời rạc của các mode còn lại -> render riêng.
            if (data.mode === 'trake_temporal') {
                displayTrakeSequences(data.results, data.events_query || [], data.max_gap_seconds);
                displaySummary(data.summary);
                statusMessage.textContent = `Tìm thấy ${data.results.length} video có chuỗi sự kiện khớp.`;
                return;
            }

            // (THÊM MỚI) Logic phân nhánh Group/Single
            const isGrouped = !Array.isArray(data.results);

            if (searchMode === 'asr') {
                if (isGrouped) {
                    displayGroupedAsrResults(data.results);
                } else {
                    displayAsrResults(data.results); // Single
                }
            } else {
                // Semantic, OCR, Similar, TRAKE Image
                if (isGrouped) {
                    displayGroupedImageResults(data.results);
                } else {
                    displayImageResults(data.results); // Single
                }
            }

            displaySummary(data.summary);
            // Đếm số lượng kết quả (phức tạp hơn)
            let totalResults = 0;
            if (isGrouped) {
                totalResults = Object.values(data.results).reduce((sum, items) => sum + items.length, 0);
            } else {
                totalResults = data.results.length;
            }
            statusMessage.textContent = `Tìm thấy ${totalResults} kết quả.`;

        } catch (error) {
            console.error('Lỗi khi tìm kiếm:', error);
            statusMessage.textContent = `Đã xảy ra lỗi: ${error.message}`;
        } finally {
            resetSearchButton();
        }
    }

    // (THÊM MỚI) 95.4 -> "95s", 132.5 -> "2p13s" cho dễ đọc khi chuỗi bị giãn
    function formatDuration(seconds) {
        const s = Math.round(seconds);
        if (s < 60) return `${s}s`;
        return `${Math.floor(s / 60)}p${String(s % 60).padStart(2, '0')}s`;
    }

    // ===== (THÊM MỚI) HIỂN THỊ KẾT QUẢ TRAKE DẠNG CHUỖI SỰ KIỆN =====
    // Mỗi video là 1 hàng gồm N ảnh theo đúng thứ tự sự kiện (đã được backend căn chỉnh thời gian),
    // kèm nút đổ thẳng cả chuỗi vào ô nộp đáp án TRAKE - không phải click từng frame như trước.
    function displayTrakeSequences(sequences, eventsQuery, maxGapSeconds) {
        imageContainer.innerHTML = '';
        if (!sequences || sequences.length === 0) {
            imageContainer.innerHTML = '<p>Không tìm thấy video nào có đủ chuỗi sự kiện.</p>';
            return;
        }

        const fragment = document.createDocumentFragment();

        sequences.forEach(seq => {
            const card = document.createElement('div');
            card.className = 'trake-seq-card';
            card.dataset.videoId = seq.videoId;

            // --- Header: video id + số sự kiện khớp + nút dùng cả chuỗi ---
            const header = document.createElement('div');
            header.className = 'trake-seq-header';

            const title = document.createElement('span');
            title.className = 'trake-seq-title';
            const spanText = seq.span_seconds !== undefined ? ` · trải dài ${formatDuration(seq.span_seconds)}` : '';
            title.textContent = `${seq.videoId} — khớp ${seq.matched_events}/${seq.total_events} sự kiện (điểm ${seq.score.toFixed(3)})${spanText}`;
            header.appendChild(title);

            const useButton = document.createElement('button');
            useButton.type = 'button';
            useButton.className = 'trake-seq-use-button';
            useButton.textContent = '⬇ Dùng chuỗi này để nộp';
            useButton.addEventListener('click', () => useTrakeSequence(seq, useButton));
            header.appendChild(useButton);

            card.appendChild(header);

            // --- Hàng ảnh theo thứ tự sự kiện ---
            const row = document.createElement('div');
            row.className = 'trake-seq-row';

            seq.events.forEach(ev => {
                const cell = document.createElement('div');
                cell.className = 'trake-seq-cell' + (ev.matched ? '' : ' trake-seq-cell-guess');

                const label = document.createElement('div');
                label.className = 'trake-seq-label';
                const queryText = eventsQuery[ev.event_index] || `Sự kiện ${ev.event_index + 1}`;
                label.textContent = `${ev.event_index + 1}. ${queryText}`;
                label.title = queryText;
                cell.appendChild(label);

                cell.appendChild(createLazyImage(ev));

                const meta = document.createElement('div');
                meta.className = 'trake-seq-meta';
                const frameLabel = ev.frame_idx !== null && ev.frame_idx !== undefined ? `frame ${ev.frame_idx}` : 'frame ?';
                meta.textContent = `${frameLabel} · ${ev.pts_time.toFixed(1)}s` + (ev.matched ? '' : ' · (đoán)');
                cell.appendChild(meta);

                // Khoảng cách tới sự kiện liền trước — tô đỏ khi vượt ngưỡng để thấy ngay chuỗi bị "giãn"
                if (ev.gap_from_prev !== null && ev.gap_from_prev !== undefined) {
                    const gapEl = document.createElement('div');
                    const tooFar = maxGapSeconds !== undefined && ev.gap_from_prev > maxGapSeconds;
                    gapEl.className = 'trake-seq-gap' + (tooFar ? ' trake-seq-gap-far' : '');
                    gapEl.textContent = `↔ cách sự kiện trước ${formatDuration(ev.gap_from_prev)}`;
                    if (tooFar) gapEl.title = `Vượt ngưỡng ${maxGapSeconds}s — chuỗi này đã bị trừ điểm`;
                    cell.appendChild(gapEl);
                }

                row.appendChild(cell);
            });

            card.appendChild(row);
            fragment.appendChild(card);
        });

        imageContainer.appendChild(fragment);
        console.log(`✅ Rendered ${sequences.length} chuỗi sự kiện TRAKE`);
    }

    // (THÊM MỚI) Đổ nguyên chuỗi N frame vào ô nộp đáp án TRAKE bằng 1 cú click
    function useTrakeSequence(seq, buttonEl) {
        const frameIds = seq.events
            .map(ev => ev.frame_idx)
            .filter(f => f !== null && f !== undefined);

        if (frameIds.length === 0) {
            alert('Chuỗi này không có frame_idx hợp lệ (thiếu metadata map-keyframes).');
            return;
        }
        if (frameIds.length < seq.events.length) {
            alert(`Cảnh báo: chỉ lấy được ${frameIds.length}/${seq.events.length} frame_idx (thiếu metadata cho một số frame).`);
        }

        // Chuyển sang chế độ nộp TRAKE và mở panel
        currentSubmissionVideoId = seq.videoId;
        document.getElementById('submit-mode-trake').checked = true;
        updateSubmissionUIVisibility();
        submissionPanel.classList.remove('hidden');

        trakeFrames = frameIds.slice().sort((a, b) => a - b);
        trakeFramesListSpan.textContent = JSON.stringify(trakeFrames);

        const originalText = buttonEl.textContent;
        buttonEl.textContent = '✓ Đã nạp vào ô nộp';
        setTimeout(() => { buttonEl.textContent = originalText; }, 1500);
    }

    // ===== (CẬP NHẬT) HÀM displayImageResults VỚI LAZY LOADING =====
    function displayImageResults(results) {
        imageContainer.innerHTML = '';
        let grid = document.getElementById('single-search-grid');
        if (!grid) {
            grid = document.createElement('div');
            grid.id = 'single-search-grid';
            imageContainer.appendChild(grid);
        }
        grid.innerHTML = '';

        if (results.length === 0) {
            grid.innerHTML = '<p>Không tìm thấy kết quả nào.</p>';
            return;
        }

        // TẠO FRAGMENT ĐỂ TỐI ƯU PERFORMANCE
        const fragment = document.createDocumentFragment();

        results.forEach(item => {
            // DÙNG HÀM LAZY LOADING
            const img = createLazyImage(item);
            fragment.appendChild(img);
        });

        // APPEND 1 LẦN (nhanh hơn append nhiều lần)
        grid.appendChild(fragment);

        console.log(`✅ Rendered ${results.length} images with Lazy Loading`);
    }

    // ===== (CẬP NHẬT) HÀM displayGroupedImageResults VỚI LAZY LOADING =====
    function displayGroupedImageResults(groups) {
        imageContainer.innerHTML = '';
        const videoIds = Object.keys(groups);
        if (videoIds.length === 0) {
            imageContainer.innerHTML = '<p>Không tìm thấy kết quả nào.</p>';
            return;
        }

        for (const videoId of videoIds) {
            const items = groups[videoId];

            const groupContainer = document.createElement('div');
            groupContainer.className = 'group-container';

            const title = document.createElement('h3');
            title.className = 'group-title';
            title.textContent = `Video: ${videoId} (${items.length} kết quả)`;
            groupContainer.appendChild(title);

            const itemsGrid = document.createElement('div');
            itemsGrid.className = 'group-items-grid';

            const fragment = document.createDocumentFragment();

            items.forEach(item => {
                // DÙNG HÀM LAZY LOADING
                const img = createLazyImage(item);
                fragment.appendChild(img);
            });

            itemsGrid.appendChild(fragment);
            groupContainer.appendChild(itemsGrid);
            imageContainer.appendChild(groupContainer);
        }

        console.log(`✅ Rendered grouped results with Lazy Loading`);
    }

    // === (CẬP NHẬT) CÁC HÀM ĐIỀU KHIỂN YOUTUBE PLAYER ===
    // 1. Tải bản đồ keyframe từ server
    // 1. Tải bản đồ keyframe từ server (ĐÃ ĐƠN GIẢN HÓA)
    async function loadKeyframeMap(internalVideoId) {
        if (internalVideoId === currentLoadedInternalMapId) return;

        console.log(`Đang tải FPS cho video: ${internalVideoId}`);
        currentLoadedInternalMapId = internalVideoId;
        clearInterval(videoTimeInterval); // Dừng interval cũ

        // (THAY ĐỔI) Reset map, chỉ cần fps
        currentKeyframeMap = { fps: null };

        currentFrameIndexSpan.textContent = "Đang tải...";
        try {
            const response = await fetch('http://localhost:5000/get_keyframe_map', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ video_id: internalVideoId })
            });
            if (!response.ok) throw new Error('Không tìm thấy bản đồ keyframe');

            const mapData = await response.json(); // Mong đợi {"fps": 25.0, ...}

            // (THAY ĐỔI) Chỉ lấy FPS.
            currentKeyframeMap.fps = mapData.fps ? parseFloat(mapData.fps) : null;

            if (currentKeyframeMap.fps) {
                console.log(`Tải thành công FPS: ${currentKeyframeMap.fps}`);
            } else {
                console.warn("Không tìm thấy giá trị 'fps' trong file map JSON.");
                currentFrameIndexSpan.textContent = "Lỗi FPS";
            }
            // === KẾT THÚC THAY ĐỔI (Đã xóa bỏ logic data, min, max, a, b) ===
        } catch (error) {
            console.error('Lỗi khi tải bản đồ keyframe:', error);
            currentFrameIndexSpan.textContent = "Lỗi";
        }
    }
    // 2. Cập nhật UI thởi gian thực (được gọi bởi setInterval)
    // 2. Cập nhật UI thởi gian thực (được gọi bởi setInterval)
    function updateRealTimeFrame() {
        if (!ytPlayer || typeof ytPlayer.getCurrentTime !== 'function') {
            return; // Player chưa sẵn sàng
        }

        const currentTime = ytPlayer.getCurrentTime();
        currentVideoTimeSpan.textContent = currentTime.toFixed(2);

        // === (THAY ĐỔI) SỬ DỤNG CÔNG THỨC Time * FPS ===
        let frameIdx = "N/A";

        // (THAY ĐỔI) Chỉ dùng công thức Time * FPS. Không kẹp, không dự phòng.
        if (currentKeyframeMap.fps !== null && currentKeyframeMap.fps > 0) {
            // Tính toán giá trị frame_idx thô (Time * FPS) và làm tròn
            frameIdx = Math.round(currentTime * currentKeyframeMap.fps);
        } else {
            // Sẽ hiển thị N/A nếu FPS chưa được tải hoặc không hợp lệ
            // (Kiểm tra này để giữ nguyên thông báo lỗi/đang tải)
            const currentStatus = currentFrameIndexSpan.textContent;
            if (currentStatus === "Đang tải..." || currentStatus.includes("Lỗi")) {
                frameIdx = currentStatus;
            }
        }

        currentFrameIndexSpan.textContent = frameIdx;
        // === KẾT THÚC THAY ĐỔI ===

        // Cập nhật UI của submission panel
        updateSubmissionUIVisibility();
    }

    // 3. Callback khi player sẵn sàng
    function onPlayerReady(event) {
        // event.target.playVideo(); // (Bỏ) Đã có trong playerVars
    }
    // 4. Callback khi trạng thái player thay đổi
    function onPlayerStateChange(event) {
        if (event.data == YT.PlayerState.PLAYING) {
            // Bắt đầu interval khi video chạy
            videoTimeInterval = setInterval(updateRealTimeFrame, 250); // Cập nhật 4 lần/giây
        } else {
            // Dừng interval khi video Tạm dừng, Kết thúc, v.v.
            clearInterval(videoTimeInterval);

            // (THÊM MỚI) Vẫn cập nhật frame một lần cuối khi dừng
            if (event.data == YT.PlayerState.PAUSED) {
                updateRealTimeFrame();
            }
        }
    }

    // === (CẬP NHẬT) Hàm hiển thị video player ===
    function showVideoPlayer(youtubeVideoId, startTime, videoTitle, internalVideoId) { // (THÊM MỚI) internalVideoId
        if (!youtubeVideoId) { console.error("Không có video ID (YouTube) để phát."); return; }
        if (!internalVideoId) { console.error("Không có ID (Internal) để tải bản đồ keyframe."); return; }

        videoPlayerTitle.textContent = videoTitle || "Video Player";
        // [SỬA LỖI !important]
        videoPlayerArea.classList.remove("hidden");

        // (THÊM MỚI) Hiển thị và reset panel submission
        // [SỬA LỖI !important]
        submissionPanel.classList.remove('hidden');
        currentSubmissionVideoId = internalVideoId; // Set ID để submit
        resetSubmissionForms(); // Hàm mới để reset
        updateSubmissionUIVisibility(); // Cập nhật video ID
        // Tải bản đồ keyframe cho video này
        loadKeyframeMap(internalVideoId); // (THAY ĐỔI) Dùng internalVideoId
        // Tạo player mới hoặc tải video mới
        if (ytPlayer && currentLoadedYoutubeId === youtubeVideoId) { // (THAY ĐỔI) Check YouTube ID
            // Nếu player đã tồn tại *và* video ID youtube giống hệt
            // Chỉ tua (seek) đến thởi gian mới
            console.log("Player tồn tại, chỉ seek to:", startTime);
            ytPlayer.seekTo(startTime, true);
            ytPlayer.playVideo(); // Đảm bảo nó phát
        } else if (ytPlayer) {
            // Nếu player đã tồn tại *nhưng* video ID khác
            console.log("Player tồn tại, tải video mới:", youtubeVideoId);
            ytPlayer.loadVideoById({
                videoId: youtubeVideoId, // ID YouTube
                startSeconds: startTime
            });
        } else {
            // Nếu chưa có player, tạo player mới
            console.log("Tạo player mới cho:", youtubeVideoId);
            ytPlayer = new YT.Player('youtube-player', { // 'youtube-player' là ID của <div>
                videoId: youtubeVideoId, // ID YouTube
                playerVars: {
                    'autoplay': 1,
                    'start': startTime
                },
                events: {
                    'onReady': onPlayerReady,
                    'onStateChange': onPlayerStateChange
                }
            });
        }
        // (THÊM MỚI) Cập nhật ID video youtube đang chạy
        currentLoadedYoutubeId = youtubeVideoId;
    }

    // (CẬP NHẬT) Hiển thị kết quả cho ASR (Single Search)
    function displayAsrResults(results) {
        asrContainer.innerHTML = ''; // Xóa group cũ
        if (results.length === 0) {
            asrContainer.innerHTML = '<p>Không tìm thấy kết quả nào.</p>';
            return;
        }

        results.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'asr-result-item';

            // (THÊM MỚI) Tạo ảnh keyframe
            if (item.web_path) {
                const img = document.createElement('img');
                img.src = item.web_path;
                img.className = 'asr-keyframe-img';
                img.title = `Time: ${item.start.toFixed(2)}s\nFrame: ${item.frame_n}`;
                img.addEventListener('click', () => {
                    // (SỬA LỖI) Tìm imgElement tương ứng
                    let galleryImg = null;
                    const galleryImages = document.querySelectorAll('.gallery-item');
                    for (let gImg of galleryImages) {
                        if (gImg.src.endsWith(item.web_path)) {
                            galleryImg = gImg;
                            break;
                        }
                    }
                    showImageDetail(item.web_path, galleryImg);
                });
                itemDiv.appendChild(img);
            }
            // (THÊM MỚI) Tạo vùng chứa content
            const contentDiv = document.createElement('div');
            contentDiv.className = 'asr-content';
            const startStr = item.start.toFixed(2); const endStr = item.end.toFixed(2);
            const startMinSec = formatTime(item.start); const endMinSec = formatTime(item.end);
            const textHtml = `<p class="asr-text">${item.text}</p>`;
            const timeString = `<span class="asr-time">${startStr} - ${endStr} (giây)</span> / <span class="asr-time-min">${startMinSec} - ${endMinSec}</span>`;
            const frame_n_display = item.frame_n !== null ? item.frame_n : "N/A";
            const frame_idx_display = item.frame_idx !== null ? item.frame_idx : "N/A";
            // (SỬA LỖI) Bỏ Video ID (đã có ở summary)
            const infoHtml = `<p class="asr-info"><strong>Frame (gần nhất):</strong> ${frame_n_display} <br><strong>Index (gần nhất):</strong> ${frame_idx_display} <br><strong>Thời gian:</strong> ${timeString}</p>`;
            contentDiv.innerHTML = textHtml + infoHtml;

            if (item.watch_url) {
                // === [SỬA LỖI VẤN ĐỀ 1] ===
                contentDiv.addEventListener('click', () => {
                    showImageDetail(item.web_path || '', item)
                });
                // === KẾT THÚC SỬA LỖI ===
            } else {
                contentDiv.style.cursor = "default";
            }

            itemDiv.appendChild(contentDiv); // Thêm content vào
            asrContainer.appendChild(itemDiv);
        });
    }

    // (CẬP NHẬT) Hiển thị kết quả ASR đã nhóm (Group Search)
    function displayGroupedAsrResults(groups) {
        asrContainer.innerHTML = '';
        const videoIds = Object.keys(groups);
        if (videoIds.length === 0) {
            asrContainer.innerHTML = '<p>Không tìm thấy kết quả nào.</p>';
            return;
        }
        for (const videoId of videoIds) {
            const items = groups[videoId]; // Đây là mảng đã được sắp xếp theo thởi gian

            // Tạo container cho nhóm
            const groupContainer = document.createElement('div');
            groupContainer.className = 'group-container'; // Dùng style chung

            // Tạo tiêu đề
            const title = document.createElement('h3');
            title.className = 'group-title';
            title.textContent = `Video: ${videoId} (${items.length} kết quả)`;
            groupContainer.appendChild(title);

            // Tạo danh sách cho các segment
            const itemsList = document.createElement('div');
            itemsList.className = 'group-items-list'; // Style riêng cho list

            items.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'asr-result-item';
                // (THÊM MỚI) Tạo ảnh keyframe
                if (item.web_path) {
                    const img = document.createElement('img');
                    img.src = item.web_path;
                    img.className = 'asr-keyframe-img';
                    img.title = `Time: ${item.start.toFixed(2)}s\nFrame: ${item.frame_n}`;
                    img.addEventListener('click', () => {
                        showImageDetail(item.web_path, null);
                    });
                    itemDiv.appendChild(img);
                }
                // (THÊM MỚI) Tạo vùng chứa content
                const contentDiv = document.createElement('div');
                contentDiv.className = 'asr-content';

                const startStr = item.start.toFixed(2); const endStr = item.end.toFixed(2);
                const startMinSec = formatTime(item.start); const endMinSec = formatTime(item.end);
                const textHtml = `<p class="asr-text">${item.text}</p>`;
                const timeString = `<span class="asr-time">${startStr} - ${endStr} (giây)</span> / <span class="asr-time-min">${startMinSec} - ${endMinSec}</span>`;
                const frame_n_display = item.frame_n !== null ? item.frame_n : "N/A";
                const frame_idx_display = item.frame_idx !== null ? item.frame_idx : "N/A";
                // (SỬA LỖI) Bỏ Video ID khỏi info, vì đã có ở tiêu đề
                const infoHtml = `<p class="asr-info"><strong>Frame (gần nhất):</strong> ${frame_n_display} <br><strong>Index (gần nhất):</strong> ${frame_idx_display} <br><strong>Thời gian:</strong> ${timeString}</p>`;
                contentDiv.innerHTML = textHtml + infoHtml;

                if (item.watch_url) {
                    // === [SỬA LỖI VẤN ĐỀ 1] ===
                    contentDiv.addEventListener('click', () => {
                        showImageDetail(item.web_path || '', item)
                    });
                    // === KẾT THÚC SỬA LỖI ===
                } else {
                    contentDiv.style.cursor = "default";
                }

                itemDiv.appendChild(contentDiv);
                itemsList.appendChild(itemDiv);
            });

            groupContainer.appendChild(itemsList);
            asrContainer.appendChild(groupContainer);
        }
    }
    // (CẬP NHẬT) Hàm displaySummary
    const SUMMARY_VISIBLE_LIMIT = 8; // (THÊM MỚI) chỉ hiện N video đầu, còn lại gấp vào nút "Xem thêm"

    function displaySummary(summary) {
        summaryBox.innerHTML = "<h4>Tóm tắt theo Video:</h4>";
        // (Sửa lỗi) summary có thể là null
        if (!summary) {
            summaryBox.innerHTML += "<p style='color: gray; font-size: 12px;'> (Không có tóm tắt)</p>";
            return;
        }

        const sortedVideoIds = Object.keys(summary).sort((a, b) => summary[b] - summary[a]);
        const validVideoIds = sortedVideoIds.filter(videoId => {
            const count = summary[videoId];
            return videoId && typeof videoId === 'string' && videoId.trim() !== '' && videoId !== 'N/A' && count && typeof count === 'number' && count > 0;
        });

        if (validVideoIds.length === 0) {
            summaryBox.innerHTML += "<p style='color: gray; font-size: 12px;'> (Không có tóm tắt)</p>";
            return;
        }

        const makeTag = (videoId) => {
            const tag = document.createElement("span");
            tag.className = "summary-tag";
            tag.textContent = `${videoId} (${summary[videoId]})`;
            tag.dataset.videoId = videoId;
            tag.addEventListener("click", () => highlightVideo(videoId));
            return tag;
        };

        validVideoIds.slice(0, SUMMARY_VISIBLE_LIMIT).forEach(videoId => {
            summaryBox.appendChild(makeTag(videoId));
        });

        const rest = validVideoIds.slice(SUMMARY_VISIBLE_LIMIT);
        if (rest.length > 0) {
            const moreButton = document.createElement("button");
            moreButton.type = "button";
            moreButton.className = "summary-more-button";
            moreButton.textContent = `Xem thêm (${rest.length})`;
            moreButton.addEventListener("click", () => {
                rest.forEach(videoId => summaryBox.insertBefore(makeTag(videoId), moreButton));
                moreButton.remove();
            });
            summaryBox.appendChild(moreButton);
        }
    }
    function highlightVideo(videoId) {
        // Logic highlight của bạn
        const tags = document.querySelectorAll(".summary-tag");
        let isHighlighting = false;
        // Kiểm tra xem tag này đã được highlight chưa
        tags.forEach(tag => {
            if (tag.dataset.videoId === videoId && tag.classList.contains('highlight')) {
                isHighlighting = true;
            }
        });
        // Tắt hết highlight
        tags.forEach(tag => tag.classList.remove('highlight'));
        // (CẬP NHẬT) Ẩn/hiện cho gallery-item (trong grid), group-container và card chuỗi TRAKE
        document.querySelectorAll('#single-search-grid .gallery-item, .group-container, .trake-seq-card').forEach(el => el.style.display = '');
        // Nếu nó chưa được highlight, thì highlight nó
        if (!isHighlighting) {
            tags.forEach(tag => {
                if (tag.dataset.videoId === videoId) {
                    tag.classList.add('highlight');
                }
            });
            // Ẩn các ảnh không khớp
            document.querySelectorAll('#single-search-grid .gallery-item').forEach(img => {
                if (img.dataset.videoId !== videoId) {
                    img.style.display = 'none';
                }
            });
            // (THÊM MỚI) Ẩn các group không khớp
            document.querySelectorAll('.group-container').forEach(group => {
                // Giả định group-title chứa videoId
                if (!group.querySelector('.group-title').textContent.includes(videoId)) {
                    group.style.display = 'none';
                }
            });
            // (THÊM MỚI) Ẩn các chuỗi TRAKE không khớp
            document.querySelectorAll('.trake-seq-card').forEach(card => {
                if (card.dataset.videoId !== videoId) {
                    card.style.display = 'none';
                }
            });
        }
    }
    // (CẬP NHẬT) Hàm showImageDetail
    async function showImageDetail(imagePath, imgElement) {
        // (SỬA LỖI) Thêm check nếu imagePath rỗng (từ ASR)
        if (!imagePath) {
            console.warn("showImageDetail được gọi với imagePath rỗng, có thể từ ASR không có keyframe.");
            // Chỉ mở video (nếu có)
            if (imgElement && imgElement.watch_url) { // imgElement ở đây là item ASR
                try {
                    const url = new URL(imgElement.watch_url);
                    const youtubeVideoId = url.searchParams.get('v');
                    const startTime = Math.floor(imgElement.start);
                    showVideoPlayer(youtubeVideoId, startTime, imgElement.video_id, imgElement.video_id);
                } catch (e) { console.error("Lỗi khi xử lý URL video ASR:", e); }
            }
            return;
        }
        document.querySelectorAll(".gallery-item.selected").forEach(el => el.classList.remove("selected"));
        if (imgElement) imgElement.classList.add("selected");
        currentDetailPath = imagePath;

        // Hiển thị panel chi tiết
        // [SỬA LỖI !important]
        detailBox.classList.remove("hidden");

        // Cập nhật thông tin panel chi tiết
        document.getElementById('detail-image').src = imagePath;
        const p = imagePath.split('/');
        const videoId_from_path = p.length > 2 ? p[p.length - 2] : "N/A"; // Thêm check
        const frameN = p.length > 1 ? p[p.length - 1].split('.')[0] : "N/A";

        document.getElementById('video-name').textContent = videoId_from_path;
        document.getElementById('meta-n').textContent = frameN !== "N/A" ? parseInt(frameN, 10) : "N/A";
        document.getElementById('meta-pts').textContent = "Đang tải...";
        document.getElementById('meta-idx').textContent = "Đang tải...";
        noVideoLinkSpan.style.display = 'inline';
        document.getElementById("neighbor-thumbnails").innerHTML = "Đang tải...";
        // Gọi API lấy metadata
        try {
            const metaRes = await fetch('http://localhost:5000/metadata', { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_path: imagePath }), });
            const meta = await metaRes.json();
            document.getElementById('meta-pts').textContent = meta.pts_time ? parseFloat(meta.pts_time).toFixed(2) : "N/A";
            document.getElementById('meta-idx').textContent = meta.frame_idx || "N/A";
            if (meta.playback_url) {
                noVideoLinkSpan.style.display = 'none';
                const url = new URL(meta.playback_url);
                const timeParam = url.searchParams.get('t');
                const startTime = timeParam ? parseInt(timeParam.replace('s', ''), 10) : 0;
                const youtubeVideoId = new URL(meta.playback_url.split('&t=')[0]).searchParams.get('v');
                const videoTitle = videoId_from_path;

                // (QUAN TRỌNG) Gọi hàm showVideoPlayer mới
                showVideoPlayer(youtubeVideoId, startTime, videoTitle, videoId_from_path); // (THAY ĐỔI) Thêm videoId_from_path
            } else {
                noVideoLinkSpan.style.display = 'inline';
                // Nếu ảnh này không có video, tắt player (nếu đang mở)
                closeVideoPlayerButton.click();
            }
        } catch (e) {
            console.error("Lỗi lấy metadata:", e);
            noVideoLinkSpan.style.display = 'inline';
            closeVideoPlayerButton.click(); // Tắt player nếu có lỗi
        }
        // Tải frame lân cận
        try {
            const neighborRes = await fetch('http://localhost:5000/neighbor_frames', { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_path: imagePath }), });
            const data = await neighborRes.json();
            currentNeighborPaths = data.neighbors || [];
            renderNeighborFrames(currentNeighborPaths, imagePath);
        } catch (e) { console.error("Lỗi lấy frame lân cận:", e); }
    }
    window.showImageDetail = showImageDetail;

    // renderNeighborFrames
    function renderNeighborFrames(neighborPaths, currentImagePath) {
        const container = document.getElementById("neighbor-thumbnails");
        container.innerHTML = "";
        neighborPaths.forEach(src => {
            const thumb = document.createElement("img");
            thumb.src = src;
            // (SỬA LỖI LOGIC) So sánh đường dẫn đầy đủ hơn
            if (currentImagePath === src) {
                thumb.classList.add("selected");
            }
            thumb.addEventListener("click", () => navigateToNeighbor(src));
            container.appendChild(thumb);
        });
        const selectedThumb = container.querySelector('img.selected');
        if (selectedThumb) { selectedThumb.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' }); }
    }
    // navigateToNeighbor
    function navigateToNeighbor(newImagePath) {
        if (newImagePath === currentDetailPath) return;
        let correspondingElement = null;
        const galleryImages = document.querySelectorAll('.gallery-item');
        // (SỬA LỖI LOGIC) So sánh đường dẫn đầy đủ
        for (let img of galleryImages) {
            // So sánh phần cuối của src, vì src trên DOM có thể là full URL
            if (img.src.endsWith(newImagePath)) {
                correspondingElement = img;
                break;
            }
        }
        showImageDetail(newImagePath, correspondingElement); // Gọi lại showImageDetail
    }
    // navigateNeighbor (phím tắt)
    function navigateNeighbor(direction) {
        if (!currentDetailPath || currentNeighborPaths.length === 0) return;
        let currentIndex = currentNeighborPaths.indexOf(currentDetailPath);
        if (currentIndex === -1) return;
        const newIndex = currentIndex + direction;
        if (newIndex >= 0 && newIndex < currentNeighborPaths.length) {
            navigateToNeighbor(currentNeighborPaths[newIndex]);
        }
    }
});