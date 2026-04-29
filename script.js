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
// ===== (THÊM MỚI) LAZY LOADING OBSERVER =====
const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            const realSrc = img.dataset.src;

            img.style.opacity = '0.3';
            img.src = realSrc;

            img.onload = () => {
                img.style.opacity = '1';
                img.classList.remove('lazy');
            };

            img.onerror = () => {
                img.style.opacity = '1';
                img.alt = 'Error loading image';
            };

            observer.unobserve(img);
        }
    });
}, {
    rootMargin: '200px',
    threshold: 0.01
});

// ===== (THÊM MỚI) HÀM TẠO ẢNH VỚI LAZY LOADING =====
function createLazyImage(item) {
    const img = document.createElement('img');

    img.dataset.src = item.path;
    img.dataset.videoId = item.videoId;
    img.className = 'gallery-item lazy';

    // Placeholder SVG
    img.src = 'https://scontent.fsgn5-12.fna.fbcdn.net/v/t39.30808-6/369264963_124044540783877_8838806758254336301_n.jpg?_nc_cat=103&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=PCtej5F_W7cQ7kNvwHbChJd&_nc_oc=Adkikv3CaSVfoxYaHoFL18swQcoiBhbhOyjArfVGM0IKwBZuw_57ipuAcd_fjReFgl4&_nc_zt=23&_nc_ht=scontent.fsgn5-12.fna&_nc_gid=_IBpOXyp9yvAhbwDu8G6jQ&oh=00_AfcoJMpvkFLQ-MUGa-9O4GqkBX4YpZkQNuXmmqlUhqy6hg&oe=690C312E';

    if (item.common_count !== undefined) {
        const label = currentSearchMode === 'trake-image' ? 'Common Images' : 'Common Parts';
        img.title = `${label}: ${item.common_count}\nTime: ${item.pts_time.toFixed(2)}s\nScore: ${item.score.toFixed(4)}`;
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
    // Sự kiện khi thay đổi chế độ tìm kiếm
    modeSemanticRadio.addEventListener('change', updateControls);
    modeOcrRadio.addEventListener('change', updateControls);
    modeAsrRadio.addEventListener('change', updateControls);
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

        qaModeUI.style.display = 'none';
        kisModeUI.style.display = 'none';
        trakeModeUI.style.display = 'none';
        if (currentSubmissionMode === 'QA') {
            qaModeUI.style.display = 'block';
        } else if (currentSubmissionMode === 'KIS') {
            kisModeUI.style.display = 'block';
        } else if (currentSubmissionMode === 'TRAKE') {
            trakeModeUI.style.display = 'block';
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
            const response = await fetch('/submit_answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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
        groupResultsBox.classList.add('hidden'); // Ẩn group box
        if (windowSizeBox) windowSizeBox.classList.add('hidden'); // Ẩn window size
        imageUploadArea.classList.add('hidden'); // Ẩn single upload mặc định
        multiImageUploadArea.classList.add('hidden'); // Ẩn multiple upload mặc định

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
        } else {
            // Các chế độ text-based: Hiện text, Ẩn uploads
            queryInput.classList.remove('hidden');
            groupResultsBox.classList.remove('hidden');
            // Chỉ hiện translate cho semantic và trake-02
            if (searchMode === 'semantic' || searchMode === 'trake-02') {
                translateBox.classList.remove('hidden');
            }
            if (searchMode === 'trake-02') {
                if (windowSizeBox) windowSizeBox.classList.remove('hidden');
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
        const query = queryInput.value.trim();

        statusMessage.textContent = 'Đang tìm kiếm...';
        imageContainer.innerHTML = ''; asrContainer.innerHTML = ''; summaryBox.innerHTML = '';

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

            fetchOptions = {
                method: 'POST',
                body: formData
                // Không set Content-Type, trình duyệt tự làm
            };
        } else if (searchMode === 'trake-image') { // (THÊM MỚI CHO TRAKE IMAGE)
            const imageFiles = multiImageUploadInput.files;
            if (imageFiles.length < 2) {
                statusMessage.textContent = 'Cần ít nhất 2 ảnh để tìm giao.';
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
        } else {
            // Các chế độ dùng text
            if (!query) {
                statusMessage.textContent = 'Vui lòng nhập nội dung tìm kiếm.';
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
            else if (searchMode === 'trake-02') { // (THÊM MỚI CHO TRAKE.02)
                endpoint = '/search_trake_02';
                payload.translate = translateCheckbox.checked; // Giữ translate cho trake-02
                if (windowSizeInput) payload.window_size = parseInt(windowSizeInput.value, 10);
            }

            fetchOptions = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            };
        }
        // (CẬP NHẬT) Fetch call dùng chung
        try {
            const response = await fetch(endpoint, fetchOptions);
            if (!response.ok) throw new Error(`Lỗi server: ${response.statusText}`);
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // (THÊM MỚI) Logic phân nhánh Group/Single
            const isGrouped = !Array.isArray(data.results);

            if (searchMode === 'asr') {
                if (isGrouped) {
                    displayGroupedAsrResults(data.results);
                } else {
                    displayAsrResults(data.results); // Single
                }
            } else {
                // Semantic, OCR, Similar, TRAKE.02, TRAKE Image
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
        }
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
            const response = await fetch('/get_keyframe_map', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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
    function displaySummary(summary) {
        summaryBox.innerHTML = "<h4>Tóm tắt theo Video:</h4>";
        // (Sửa lỗi) summary có thể là null
        if (!summary) {
            summaryBox.innerHTML += "<p style='color: gray; font-size: 12px;'> (Không có tóm tắt)</p>";
            return;
        }

        const sortedVideoIds = Object.keys(summary).sort((a, b) => summary[b] - summary[a]);
        if (sortedVideoIds.length === 0) {
            summaryBox.innerHTML += "<p style='color: gray; font-size: 12px;'> (Không có tóm tắt)</p>";
            return;
        }
        for (const videoId of sortedVideoIds) {
            const count = summary[videoId];
            if (videoId && typeof videoId === 'string' && videoId.trim() !== '' && videoId !== 'N/A' && count && typeof count === 'number' && count > 0) {
                const tag = document.createElement("span");
                tag.className = "summary-tag";
                tag.textContent = `${videoId} (${count})`;
                tag.dataset.videoId = videoId;
                tag.addEventListener("click", () => highlightVideo(videoId));
                summaryBox.appendChild(tag);
            }
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
        // (CẬP NHẬT) Ẩn/hiện cho cả gallery-item (trong grid) và group-container
        document.querySelectorAll('#single-search-grid .gallery-item, .group-container').forEach(el => el.style.display = '');
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
            const metaRes = await fetch('/metadata', { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_path: imagePath }), });
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
            const neighborRes = await fetch('/neighbor_frames', { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_path: imagePath }), });
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