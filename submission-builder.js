(function () {
    'use strict';

    const API_BASE_URL = window.location.protocol === 'file:'
        ? 'http://localhost:5000'
        : window.location.origin;

    const elements = {
        newQueryId: document.getElementById('new-query-id'),
        newEventCount: document.getElementById('new-event-count'),
        eventCountRow: document.getElementById('event-count-row'),
        addQuery: document.getElementById('add-query'),
        importQueryFiles: document.getElementById('import-query-files'),
        queryList: document.getElementById('query-list'),
        queryCount: document.getElementById('query-count'),
        emptyWorkspace: document.getElementById('empty-workspace'),
        queryWorkspace: document.getElementById('query-workspace'),
        activeQueryType: document.getElementById('active-query-type'),
        activeQueryTitle: document.getElementById('active-query-title'),
        activeQueryStats: document.getElementById('active-query-stats'),
        queryPrompt: document.getElementById('query-prompt'),
        qaAnswerRow: document.getElementById('qa-answer-row'),
        qaAnswer: document.getElementById('qa-answer'),
        answerLength: document.getElementById('answer-length'),
        trakeEventRow: document.getElementById('trake-event-row'),
        activeEventCount: document.getElementById('active-event-count'),
        manualCandidate: document.getElementById('manual-candidate'),
        manualFormatHint: document.getElementById('manual-format-hint'),
        addManualCandidate: document.getElementById('add-manual-candidate'),
        deleteQuery: document.getElementById('delete-query'),
        fillResults: document.getElementById('fill-results'),
        neighborFillControl: document.getElementById('neighbor-fill-control'),
        neighborTimeBorder: document.getElementById('neighbor-time-border'),
        clearAutomatic: document.getElementById('clear-automatic'),
        previewCsv: document.getElementById('preview-csv'),
        rankingRows: document.getElementById('ranking-rows'),
        finalCount: document.getElementById('final-count'),
        builderMessage: document.getElementById('builder-message'),
        csvPreviewPanel: document.getElementById('csv-preview-panel'),
        csvPreview: document.getElementById('csv-preview'),
        closePreview: document.getElementById('close-preview'),
        exportZip: document.getElementById('export-zip'),
        exportProject: document.getElementById('export-project'),
        importProject: document.getElementById('import-project'),
        importSubmissionFolder: document.getElementById('import-submission-folder'),
        frameDetailModal: document.getElementById('frame-detail-modal'),
        frameDetailQuery: document.getElementById('frame-detail-query'),
        frameDetailTitle: document.getElementById('frame-detail-title'),
        frameDetailEvents: document.getElementById('frame-detail-events'),
        frameDetailLoading: document.getElementById('frame-detail-loading'),
        frameDetailContent: document.getElementById('frame-detail-content'),
        frameDetailPlayer: document.getElementById('frame-detail-player'),
        frameDetailLocalPlayer: document.getElementById('frame-detail-local-player'),
        frameDetailVideoMissing: document.getElementById('frame-detail-video-missing'),
        frameDetailImage: document.getElementById('frame-detail-image'),
        frameDetailVideoId: document.getElementById('frame-detail-video-id'),
        frameDetailFrameN: document.getElementById('frame-detail-frame-n'),
        frameDetailFrameIdx: document.getElementById('frame-detail-frame-idx'),
        frameDetailTime: document.getElementById('frame-detail-time'),
        frameDetailOpenVideo: document.getElementById('frame-detail-open-video'),
        closeFrameDetail: document.getElementById('close-frame-detail'),
    };

    function message(text, isError = false) {
        elements.builderMessage.textContent = text || '';
        elements.builderMessage.classList.toggle('error', isError);
    }

    function activeQuery(state = PrelimSubmission.load()) {
        return state.queries[state.activeQueryId] || null;
    }

    function normalizeQueryId(value) {
        return String(value || '').trim().replace(/\.(txt|csv)$/i, '');
    }

    function validatedQueryType(queryId) {
        const match = String(queryId || '').match(/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}-(kis|qa|trake)$/i);
        return match ? match[1].toLowerCase() : '';
    }

    function normalizeEventCount(value) {
        const count = Number(value);
        return Number.isFinite(count) ? Math.max(2, Math.min(30, Math.trunc(count))) : 2;
    }

    function createQuery(queryId, prompt = '') {
        const id = normalizeQueryId(queryId);
        const type = validatedQueryType(id);
        if (!type) throw new Error('Tên query phải kết thúc bằng -kis, -qa hoặc -trake.');
        const state = PrelimSubmission.load();
        const duplicate = Object.keys(state.queries).find(key => key.toLowerCase() === id.toLowerCase());
        if (duplicate) {
            state.activeQueryId = duplicate;
            if (prompt && !state.queries[duplicate].prompt) state.queries[duplicate].prompt = prompt;
            PrelimSubmission.save(state);
            return;
        }
        state.queries[id] = {
            id,
            type,
            eventCount: type === 'trake'
                ? normalizeEventCount(state.latestPools?.trakeEventCount || elements.newEventCount.value)
                : 0,
            prompt,
            answer: '',
            neighborTimeBorder: 30,
            pinned: [],
            pool: type === 'trake'
                ? (state.latestPools?.trake || []).slice()
                : (state.latestPools?.frame || []).slice(),
            automatic: [],
        };
        state.activeQueryId = id;
        PrelimSubmission.save(state);
    }

    function candidateDisplay(query, candidate) {
        if (query.type === 'trake') {
            return [candidate.videoId, ...(candidate.frameIndices || [])].join(',');
        }
        const base = `${candidate.videoId},${candidate.frameIdx}`;
        if (query.type === 'qa') {
            const answer = candidate.answer || query.answer;
            return `${base},${answer ? csvCell(answer) : '<thiếu answer>'}`;
        }
        return base;
    }

    function candidateThumbnailPath(query, candidate) {
        if (query.type === 'trake') return (candidate.paths || [])[0] || '';
        return candidate.path || '';
    }

    function csvCell(value) {
        const text = String(value ?? '');
        return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    }

    function csvPreviewFor(query) {
        return PrelimSubmission.finalRows(query).map(candidate => {
            if (query.type === 'trake') {
                return [candidate.videoId, ...(candidate.frameIndices || [])].map(csvCell).join(',');
            }
            const cells = [candidate.videoId, candidate.frameIdx];
            if (query.type === 'qa') cells.push(query.answer || '');
            return cells.map(csvCell).join(',');
        }).join('\n');
    }

    function saveQueryField(field, value) {
        const state = PrelimSubmission.load();
        const query = activeQuery(state);
        if (!query) return;
        query[field] = value;
        PrelimSubmission.save(state);
    }

    function renderQueryList(state) {
        elements.queryList.textContent = '';
        const queries = Object.values(state.queries)
            .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
        elements.queryCount.textContent = String(queries.length);
        if (!queries.length) {
            const empty = document.createElement('p');
            empty.className = 'query-list-empty';
            empty.textContent = 'Chưa có query nào.';
            elements.queryList.appendChild(empty);
            return;
        }
        for (const query of queries) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = `query-item${query.id === state.activeQueryId ? ' active' : ''}`;
            const label = document.createElement('span');
            label.textContent = query.id;
            const detail = document.createElement('small');
            detail.textContent = `${query.type.toUpperCase()} · ${(query.pinned || []).length} ghim · ${(query.automatic || []).length} auto`;
            label.appendChild(detail);
            const count = document.createElement('span');
            count.className = 'query-item-count';
            count.textContent = String(PrelimSubmission.finalRows(query).length);
            button.append(label, count);
            button.addEventListener('click', () => {
                const latest = PrelimSubmission.load();
                latest.activeQueryId = query.id;
                PrelimSubmission.save(latest);
            });
            elements.queryList.appendChild(button);
        }
    }

    function actionButton(label, title, handler) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = label;
        button.title = title;
        button.addEventListener('click', handler);
        return button;
    }

    function mutatePinned(index, action) {
        const state = PrelimSubmission.load();
        const query = activeQuery(state);
        if (!query) return;
        if (action === 'remove') query.pinned.splice(index, 1);
        if (action === 'up' && index > 0) {
            [query.pinned[index - 1], query.pinned[index]] = [query.pinned[index], query.pinned[index - 1]];
        }
        if (action === 'down' && index < query.pinned.length - 1) {
            [query.pinned[index + 1], query.pinned[index]] = [query.pinned[index], query.pinned[index + 1]];
        }
        PrelimSubmission.save(state);
    }

    function reorderPinned(fromIndex, targetIndex, placeAfter = false) {
        const state = PrelimSubmission.load();
        const query = activeQuery(state);
        if (!query) return;
        const pinned = query.pinned || [];
        if (fromIndex < 0 || fromIndex >= pinned.length
                || targetIndex < 0 || targetIndex >= pinned.length) return;

        let insertionIndex = targetIndex + (placeAfter ? 1 : 0);
        const [moved] = pinned.splice(fromIndex, 1);
        if (fromIndex < insertionIndex) insertionIndex -= 1;
        pinned.splice(Math.max(0, Math.min(insertionIndex, pinned.length)), 0, moved);
        query.pinned = pinned;
        PrelimSubmission.save(state);
    }

    function clearDragIndicators() {
        elements.rankingRows.querySelectorAll?.('.drag-before, .drag-after, .dragging')
            .forEach(row => row.classList.remove('drag-before', 'drag-after', 'dragging'));
    }

    function removeAutomatic(index) {
        const state = PrelimSubmission.load();
        const query = activeQuery(state);
        if (!query) return;
        query.automatic.splice(index, 1);
        PrelimSubmission.save(state);
    }

    function parseManualCandidate(query, rawValue) {
        const parts = String(rawValue || '').trim().split(/[\s,;]+/).filter(Boolean);
        const expectedFrames = query.type === 'trake' ? normalizeEventCount(query.eventCount) : 1;
        if (parts.length !== expectedFrames + 1) {
            const expected = query.type === 'trake'
                ? `${expectedFrames} frame_idx theo đúng thứ tự event`
                : 'một frame_idx';
            throw new Error(`Cần video_id và ${expected}.`);
        }

        const videoId = parts[0].toUpperCase();
        if (!/^L\d{2}_V\d+$/.test(videoId)) {
            throw new Error('video_id không hợp lệ. Ví dụ: L22_V013.');
        }
        const frameIndices = parts.slice(1).map(value => Number(value));
        if (frameIndices.some(value => !Number.isSafeInteger(value) || value < 0)) {
            throw new Error('frame_idx phải là số nguyên không âm.');
        }

        return query.type === 'trake'
            ? { videoId, frameIndices, paths: [] }
            : { videoId, frameIdx: frameIndices[0], path: '' };
    }

    async function hydrateManualCandidate(query, candidate) {
        const frameIndices = query.type === 'trake'
            ? candidate.frameIndices
            : [candidate.frameIdx];
        try {
            const paths = await Promise.all(frameIndices.map(async frameIdx => {
                const response = await fetch(`${API_BASE_URL}/submission/playback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ videoId: candidate.videoId, frameIdx }),
                });
                const data = await response.json();
                return response.ok && !data.error ? (data.path || '') : '';
            }));
            if (query.type === 'trake') candidate.paths = paths;
            else candidate.path = paths[0] || '';
        } catch (_) {
            // Không có thumbnail vẫn lưu được dòng; nút Xem sẽ resolve lại khi bấm.
        }
        return candidate;
    }

    async function addManualCandidate() {
        const state = PrelimSubmission.load();
        const query = activeQuery(state);
        if (!query) return;
        if ((query.pinned || []).length >= 100) {
            message('Query đã đủ 100 dòng ghim thủ công.', true);
            return;
        }

        const oldText = elements.addManualCandidate.textContent;
        elements.addManualCandidate.disabled = true;
        elements.addManualCandidate.textContent = 'Đang thêm...';
        try {
            const candidate = parseManualCandidate(query, elements.manualCandidate.value);
            const key = PrelimSubmission.candidateKey(query.type, candidate);
            if ((query.pinned || []).some(item => PrelimSubmission.candidateKey(query.type, item) === key)) {
                throw new Error('Dòng này đã có trong phần ghim thủ công.');
            }

            await hydrateManualCandidate(query, candidate);
            query.pinned = [...(query.pinned || []), candidate];
            query.automatic = (query.automatic || [])
                .filter(item => PrelimSubmission.candidateKey(query.type, item) !== key);
            PrelimSubmission.save(state);
            elements.manualCandidate.value = '';
            message(`Đã ghim thủ công ${candidateDisplay(query, candidate)}.`);
        } catch (error) {
            message(`Không thêm được: ${error.message}`, true);
        } finally {
            elements.addManualCandidate.disabled = false;
            elements.addManualCandidate.textContent = oldText;
        }
    }

    async function viewCandidate(query, candidate, button) {
        const frameIdx = query.type === 'trake'
            ? (candidate.frameIndices || [])[0]
            : candidate.frameIdx;
        const keyframePath = query.type === 'trake'
            ? (candidate.paths || [])[0]
            : candidate.path;
        if (frameIdx === undefined || frameIdx === null) {
            message('Dòng này chưa có frame_idx để mở video.', true);
            return;
        }

        // Mở tab ngay trong click event để trình duyệt không chặn popup sau await.
        const previewTab = window.open('about:blank', '_blank');
        if (!previewTab) {
            message('Trình duyệt đang chặn popup. Hãy cho phép popup cho trang này.', true);
            return;
        }
        previewTab.document.title = 'Đang mở video...';
        previewTab.document.body.textContent = 'Đang tìm timestamp gần nhất...';
        const oldText = button.textContent;
        button.disabled = true;
        button.textContent = '...';
        try {
            // KIS/QA và TRAKE mới đều giữ path keyframe: dùng API metadata cũ,
            // không phụ thuộc server đã restart để nhận route playback mới.
            const hasExactTimestamp = candidate.ptsTime !== undefined && candidate.ptsTime !== null
                && candidate.ptsTime !== '' && Number.isFinite(Number(candidate.ptsTime));
            const endpoint = keyframePath && !hasExactTimestamp ? '/metadata' : '/submission/playback';
            const requestBody = keyframePath && !hasExactTimestamp
                ? { image_path: keyframePath }
                : { videoId: candidate.videoId, frameIdx, ptsTime: candidate.ptsTime };
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(data.error || 'Không tìm thấy video.');
            let target = data.playback_url || data.path;
            if (!target) throw new Error('Video này không có playback URL hoặc keyframe fallback.');
            if (data.playback_type === 'local') {
                const localTarget = new URL(target, window.location.origin);
                localTarget.hash = `t=${Math.max(0, Number(data.playback_start ?? data.pts_time ?? 0) || 0)}`;
                target = localTarget.href;
            }
            previewTab.location.href = new URL(target, window.location.origin).href;
        } catch (error) {
            previewTab.close();
            message(`Không mở được video: ${error.message}`, true);
        } finally {
            button.disabled = false;
            button.textContent = oldText;
        }
    }

    function youtubeEmbedUrl(playbackUrl, fallbackSeconds = 0) {
        if (!playbackUrl) return '';
        try {
            const url = new URL(playbackUrl, window.location.origin);
            let videoId = '';
            if (url.hostname.includes('youtu.be')) videoId = url.pathname.split('/').filter(Boolean)[0] || '';
            else if (url.hostname.includes('youtube.com')) {
                videoId = url.searchParams.get('v') || url.pathname.match(/\/embed\/([^/?]+)/)?.[1] || '';
            }
            if (!videoId) return '';
            const rawStart = url.searchParams.get('t') || url.searchParams.get('start') || fallbackSeconds;
            const start = Math.max(0, parseInt(String(rawStart).replace(/s$/i, ''), 10) || 0);
            return `https://www.youtube.com/embed/${encodeURIComponent(videoId)}?autoplay=1&start=${start}`;
        } catch (_) {
            return '';
        }
    }

    function closeFrameDetail() {
        elements.frameDetailModal.classList.add('hidden');
        elements.frameDetailPlayer.src = '';
        elements.frameDetailLocalPlayer.pause();
        elements.frameDetailLocalPlayer.removeAttribute('src');
        elements.frameDetailLocalPlayer.load();
    }

    async function loadFrameDetail(query, candidate, eventIndex = 0) {
        const isTrake = query.type === 'trake';
        const frameIdx = isTrake ? (candidate.frameIndices || [])[eventIndex] : candidate.frameIdx;
        const keyframePath = isTrake ? (candidate.paths || [])[eventIndex] : candidate.path;
        elements.frameDetailLoading.textContent = 'Đang tải metadata...';
        elements.frameDetailLoading.classList.remove('hidden');
        elements.frameDetailContent.classList.add('hidden');
        elements.frameDetailPlayer.src = '';
        elements.frameDetailPlayer.classList.add('hidden');
        elements.frameDetailLocalPlayer.pause();
        elements.frameDetailLocalPlayer.removeAttribute('src');
        elements.frameDetailLocalPlayer.load();
        elements.frameDetailLocalPlayer.classList.add('hidden');

        Array.from(elements.frameDetailEvents.children).forEach((button, index) => {
            button.classList.toggle('active', index === eventIndex);
        });

        try {
            let response;
            const hasExactTimestamp = candidate.ptsTime !== undefined && candidate.ptsTime !== null
                && candidate.ptsTime !== '' && Number.isFinite(Number(candidate.ptsTime));
            if (keyframePath && !hasExactTimestamp) {
                response = await fetch(`${API_BASE_URL}/metadata`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_path: keyframePath }),
                });
            } else {
                response = await fetch(`${API_BASE_URL}/submission/playback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ videoId: candidate.videoId, frameIdx, ptsTime: candidate.ptsTime }),
                });
            }
            const meta = await response.json();
            if (!response.ok || meta.error) throw new Error(meta.error || 'Không đọc được metadata.');

            const imagePath = keyframePath || meta.path || '';
            const playbackUrl = meta.playback_url || '';
            const ptsTime = Number(meta.pts_time || 0);
            const embedUrl = youtubeEmbedUrl(playbackUrl, ptsTime);
            const isLocalPlayback = meta.playback_type === 'local'
                || (playbackUrl && new URL(playbackUrl, window.location.origin).pathname.startsWith('/videos/'));
            elements.frameDetailTitle.textContent = candidate.videoId;
            elements.frameDetailVideoId.textContent = candidate.videoId;
            elements.frameDetailFrameN.textContent = meta.n ?? 'N/A';
            elements.frameDetailFrameIdx.textContent = frameIdx ?? meta.frameIdx ?? meta.frame_idx ?? 'N/A';
            elements.frameDetailTime.textContent = `${ptsTime.toFixed(2)} giây`;
            elements.frameDetailImage.src = imagePath
                ? new URL(imagePath, `${API_BASE_URL}/`).href
                : '';
            elements.frameDetailImage.classList.toggle('hidden', !imagePath);
            if (isLocalPlayback) {
                const absolutePlaybackUrl = new URL(playbackUrl, `${API_BASE_URL}/`).href;
                elements.frameDetailLocalPlayer.classList.remove('hidden');
                elements.frameDetailLocalPlayer.src = absolutePlaybackUrl;
                elements.frameDetailLocalPlayer.addEventListener('loadedmetadata', () => {
                    elements.frameDetailLocalPlayer.currentTime = Math.min(
                        Math.max(0, Number(meta.playback_start ?? ptsTime) || 0),
                        Number.isFinite(elements.frameDetailLocalPlayer.duration)
                            ? elements.frameDetailLocalPlayer.duration
                            : Math.max(0, Number(meta.playback_start ?? ptsTime) || 0)
                    );
                    elements.frameDetailLocalPlayer.play().catch(() => {});
                }, { once: true });
                elements.frameDetailLocalPlayer.load();
            } else {
                elements.frameDetailPlayer.classList.toggle('hidden', !embedUrl);
                elements.frameDetailPlayer.src = embedUrl;
            }
            elements.frameDetailVideoMissing.classList.toggle('hidden', Boolean(isLocalPlayback || embedUrl));
            const openVideoUrl = playbackUrl
                ? new URL(playbackUrl, `${API_BASE_URL}/`)
                : null;
            if (openVideoUrl && isLocalPlayback) {
                openVideoUrl.hash = `t=${Math.max(0, Number(meta.playback_start ?? ptsTime) || 0)}`;
            }
            elements.frameDetailOpenVideo.href = openVideoUrl ? openVideoUrl.href : '#';
            elements.frameDetailOpenVideo.classList.toggle('hidden', !playbackUrl);
            elements.frameDetailLoading.classList.add('hidden');
            elements.frameDetailContent.classList.remove('hidden');
        } catch (error) {
            elements.frameDetailLoading.textContent = `Không tải được thông tin frame: ${error.message}`;
        }
    }

    function showFrameDetail(query, candidate) {
        elements.frameDetailQuery.textContent = `${query.type.toUpperCase()} · ${query.id}`;
        elements.frameDetailTitle.textContent = candidate.videoId || 'Thông tin frame';
        elements.frameDetailEvents.textContent = '';
        const frameIndices = query.type === 'trake'
            ? (candidate.frameIndices || [])
            : [candidate.frameIdx];
        if (frameIndices.length > 1) {
            elements.frameDetailEvents.classList.remove('hidden');
            frameIndices.forEach((frameIdx, index) => {
                const button = actionButton(
                    `Event ${index + 1} · ${frameIdx}`,
                    `Xem frame của event ${index + 1}`,
                    () => loadFrameDetail(query, candidate, index)
                );
                elements.frameDetailEvents.appendChild(button);
            });
        } else {
            elements.frameDetailEvents.classList.add('hidden');
        }
        elements.frameDetailModal.classList.remove('hidden');
        loadFrameDetail(query, candidate, 0);
    }

    function renderRanking(query) {
        elements.rankingRows.textContent = '';
        const finalRows = PrelimSubmission.finalRows(query);
        elements.finalCount.textContent = `${finalRows.length}/100`;
        if (!finalRows.length) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 4;
            td.className = 'empty-ranking';
            td.textContent = 'Chưa có kết quả. Hãy ghim ở trang search hoặc fill từ ranking.';
            tr.appendChild(td);
            elements.rankingRows.appendChild(tr);
            return;
        }

        const pinnedCount = PrelimSubmission.dedupe(query.type, query.pinned || []).length;
        finalRows.forEach((candidate, index) => {
            const isPinned = index < pinnedCount;
            const tr = document.createElement('tr');
            if (isPinned) {
                tr.className = 'pinned-row';
                tr.dataset.pinnedIndex = String(index);
                tr.addEventListener('dragover', event => {
                    event.preventDefault();
                    const bounds = tr.getBoundingClientRect();
                    const placeAfter = event.clientY >= bounds.top + bounds.height / 2;
                    tr.classList.toggle('drag-before', !placeAfter);
                    tr.classList.toggle('drag-after', placeAfter);
                    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
                });
                tr.addEventListener('dragleave', () => {
                    tr.classList.remove('drag-before', 'drag-after');
                });
                tr.addEventListener('drop', event => {
                    event.preventDefault();
                    const fromIndex = Number(event.dataTransfer?.getData('text/plain'));
                    const bounds = tr.getBoundingClientRect();
                    const placeAfter = event.clientY >= bounds.top + bounds.height / 2;
                    clearDragIndicators();
                    if (Number.isInteger(fromIndex)) reorderPinned(fromIndex, index, placeAfter);
                });
            }

            const rankCell = document.createElement('td');
            rankCell.textContent = String(index + 1);
            const sourceCell = document.createElement('td');
            sourceCell.className = 'source-badge';
            sourceCell.textContent = isPinned ? '📌 Thủ công' : '⚙ Auto';
            const valueCell = document.createElement('td');
            valueCell.className = 'row-value';
            const resultWrap = document.createElement('div');
            resultWrap.className = 'result-with-thumbnail';
            const thumbnailPath = candidateThumbnailPath(query, candidate);
            if (thumbnailPath) {
                const thumbnail = document.createElement('img');
                thumbnail.className = 'ranking-thumbnail';
                thumbnail.src = new URL(thumbnailPath, `${API_BASE_URL}/`).href;
                thumbnail.alt = `Keyframe ${candidate.videoId}`;
                thumbnail.loading = 'lazy';
                thumbnail.title = 'Bấm để xem thông tin frame và video';
                resultWrap.appendChild(thumbnail);
            } else {
                const placeholder = document.createElement('span');
                placeholder.className = 'ranking-thumbnail thumbnail-missing';
                placeholder.textContent = 'No image';
                resultWrap.appendChild(placeholder);
            }
            const valueText = document.createElement('span');
            valueText.textContent = candidateDisplay(query, candidate);
            resultWrap.appendChild(valueText);
            valueCell.appendChild(resultWrap);
            const actionsCell = document.createElement('td');
            actionsCell.className = 'row-actions';
            const watchButton = actionButton('▶ Xem', 'Mở video tại frame này', () => {
                viewCandidate(query, candidate, watchButton);
            });
            const thumbnail = resultWrap.querySelector('img.ranking-thumbnail');
            if (thumbnail) thumbnail.addEventListener('click', () => showFrameDetail(query, candidate));
            actionsCell.appendChild(watchButton);
            if (isPinned) {
                const dragHandle = document.createElement('span');
                dragHandle.className = 'drag-handle';
                dragHandle.textContent = '⠿';
                dragHandle.title = 'Kéo để đổi thứ tự; dùng phím ↑ ↓ khi đang focus';
                dragHandle.setAttribute('role', 'button');
                dragHandle.setAttribute('tabindex', '0');
                dragHandle.setAttribute('aria-label', `Đổi vị trí dòng ${index + 1}`);
                dragHandle.draggable = true;
                dragHandle.addEventListener('dragstart', event => {
                    event.dataTransfer.setData('text/plain', String(index));
                    event.dataTransfer.effectAllowed = 'move';
                    tr.classList.add('dragging');
                });
                dragHandle.addEventListener('dragend', clearDragIndicators);
                dragHandle.addEventListener('keydown', event => {
                    if (event.key === 'ArrowUp') {
                        event.preventDefault();
                        mutatePinned(index, 'up');
                    } else if (event.key === 'ArrowDown') {
                        event.preventDefault();
                        mutatePinned(index, 'down');
                    }
                });
                actionsCell.prepend(dragHandle);
                if (query.type !== 'trake') {
                    const aroundButton = actionButton('◎ Fill quanh', 'Auto-fill quanh đúng video và timestamp này', () => {
                        fillAroundAnchors([candidate], aroundButton);
                    });
                    actionsCell.appendChild(aroundButton);
                }
                actionsCell.append(
                    actionButton('×', 'Bỏ ghim', () => mutatePinned(index, 'remove')),
                );
            } else {
                actionsCell.append(actionButton('×', 'Bỏ dòng auto', () => removeAutomatic(index - pinnedCount)));
            }
            tr.append(rankCell, sourceCell, valueCell, actionsCell);
            elements.rankingRows.appendChild(tr);
        });
    }

    function render() {
        const state = PrelimSubmission.load();
        renderQueryList(state);
        const query = activeQuery(state);
        elements.emptyWorkspace.classList.toggle('hidden', Boolean(query));
        elements.queryWorkspace.classList.toggle('hidden', !query);
        if (!query) return;

        elements.activeQueryType.textContent = query.type.toUpperCase();
        elements.activeQueryTitle.textContent = query.id;
        elements.activeQueryStats.textContent =
            `${(query.pinned || []).length} ghim · ${(query.automatic || []).length} frame lân cận auto-fill`;
        elements.queryPrompt.value = query.prompt || '';
        elements.qaAnswerRow.classList.toggle('hidden', query.type !== 'qa');
        elements.qaAnswer.value = query.answer || '';
        elements.answerLength.textContent = `${(query.answer || '').length}/100`;
        elements.trakeEventRow.classList.toggle('hidden', query.type !== 'trake');
        elements.activeEventCount.value = query.eventCount || 2;
        elements.neighborFillControl.classList.toggle('hidden', query.type === 'trake');
        elements.fillResults.classList.toggle('hidden', query.type === 'trake');
        elements.neighborTimeBorder.value = Number(query.neighborTimeBorder || 30);
        if (query.type === 'trake') {
            const count = normalizeEventCount(query.eventCount);
            elements.manualCandidate.placeholder = `L22_V013,${Array.from({ length: count }, (_, i) => 18816 + i * 30).join(',')}`;
            elements.manualFormatHint.textContent = `TRAKE: video_id và đúng ${count} frame_idx theo thứ tự event.`;
        } else {
            elements.manualCandidate.placeholder = 'L22_V013,18816';
            elements.manualFormatHint.textContent = `${query.type.toUpperCase()}: video_id,frame_idx`;
        }
        renderRanking(query);
        if (!elements.csvPreviewPanel.classList.contains('hidden')) {
            elements.csvPreview.value = csvPreviewFor(query);
        }
    }

    async function fillAroundAnchors(anchors, triggerButton = elements.fillResults) {
        const state = PrelimSubmission.load();
        const query = activeQuery(state);
        if (!query) return;
        if (query.type === 'trake') {
            message('Auto-fill lân cận hiện chỉ áp dụng cho KIS và QA.', true);
            return;
        }
        const usableAnchors = PrelimSubmission.dedupe(query.type, anchors || [])
            .filter(item => item.videoId && item.frameIdx !== undefined && item.frameIdx !== null);
        if (!usableAnchors.length) {
            message('Hãy ghim ít nhất một frame trước khi auto-fill lân cận.', true);
            return;
        }

        const timeBorder = Math.max(1, Math.min(600, Number(elements.neighborTimeBorder.value) || 30));
        query.neighborTimeBorder = timeBorder;
        const oldText = triggerButton.textContent;
        triggerButton.disabled = true;
        triggerButton.textContent = 'Đang lấy frame quanh mốc...';
        try {
            const response = await fetch(`${API_BASE_URL}/submission/neighbors`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    anchors: usableAnchors.map(item => ({
                        videoId: item.videoId,
                        frameIdx: item.frameIdx,
                        ptsTime: item.ptsTime,
                    })),
                    timeBorder,
                    limit: 1000,
                }),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || 'Không lấy được frame lân cận.');
            }

            const pinned = PrelimSubmission.dedupe(query.type, query.pinned || []);
            const pinnedKeys = new Set(pinned.map(item => PrelimSubmission.candidateKey(query.type, item)));
            const needed = Math.max(0, 100 - pinned.length);
            query.automatic = PrelimSubmission.dedupe(query.type, data.results || [])
                .filter(item => !pinnedKeys.has(PrelimSubmission.candidateKey(query.type, item)))
                .slice(0, needed);
            PrelimSubmission.save(state);
            message(
                `Đã giữ ${pinned.length} dòng ghim và fill ${query.automatic.length} frame `
                + `trong ±${timeBorder}s quanh ${usableAnchors.length} mốc đã chọn.`
            );
        } catch (error) {
            message(`Không auto-fill được: ${error.message}`, true);
        } finally {
            triggerButton.disabled = false;
            triggerButton.textContent = oldText;
        }
    }

    async function fillFromPinned() {
        const query = activeQuery();
        if (!query) return;
        await fillAroundAnchors(query.pinned || []);
    }

    function exportPayload(state) {
        return {
            queries: Object.values(state.queries).map(query => ({
                id: query.id,
                type: query.type,
                eventCount: Number(query.eventCount || 0),
                rows: PrelimSubmission.finalRows(query).map(candidate => {
                    if (query.type === 'trake') {
                        return {
                            videoId: candidate.videoId,
                            frameIndices: candidate.frameIndices,
                            paths: candidate.paths,
                        };
                    }
                    return {
                        videoId: candidate.videoId,
                        frameIdx: candidate.frameIdx,
                        ...(query.type === 'qa' ? { answer: candidate.answer || query.answer || '' } : {}),
                    };
                }),
            })),
        };
    }

    async function exportZip() {
        const state = PrelimSubmission.load();
        if (!Object.keys(state.queries).length) {
            message('Chưa có query nào để xuất.', true);
            return;
        }
        elements.exportZip.disabled = true;
        const oldText = elements.exportZip.textContent;
        elements.exportZip.textContent = 'Đang kiểm tra...';
        try {
            const response = await fetch(`${API_BASE_URL}/submission/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(exportPayload(state)),
            });
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Không tạo được ZIP.');
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'submission.zip';
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            message('Đã tạo submission.zip và kiểm tra toàn bộ CSV.');
        } catch (error) {
            message(error.message, true);
        } finally {
            elements.exportZip.disabled = false;
            elements.exportZip.textContent = oldText;
        }
    }

    function downloadJson(filename, value) {
        const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    function parseCsv(text) {
        const rows = [];
        let row = [];
        let field = '';
        let inQuotes = false;
        const source = String(text || '').replace(/^\uFEFF/, '');

        const finishRow = () => {
            row.push(field);
            if (row.some(value => value.trim() !== '')) rows.push(row);
            row = [];
            field = '';
        };

        for (let index = 0; index < source.length; index += 1) {
            const character = source[index];
            if (inQuotes) {
                if (character === '"') {
                    if (source[index + 1] === '"') {
                        field += '"';
                        index += 1;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    field += character;
                }
                continue;
            }
            if (character === '"' && field === '') {
                inQuotes = true;
            } else if (character === ',') {
                row.push(field);
                field = '';
            } else if (character === '\n') {
                finishRow();
            } else if (character !== '\r') {
                field += character;
            }
        }
        if (inQuotes) throw new Error('CSV có dấu nháy kép chưa đóng.');
        if (field !== '' || row.length) finishRow();
        return rows;
    }

    function parseFrameIndex(value, fileName, rowNumber) {
        const normalized = String(value || '').trim();
        if (!/^\d+$/.test(normalized) || !Number.isSafeInteger(Number(normalized))) {
            throw new Error(`${fileName}, dòng ${rowNumber}: frame_idx "${normalized}" không hợp lệ.`);
        }
        return Number(normalized);
    }

    function parseSubmissionCsv(fileName, text) {
        const id = normalizeQueryId(fileName);
        const type = validatedQueryType(id);
        if (!type) {
            throw new Error(`${fileName}: tên file phải kết thúc bằng -kis.csv, -qa.csv hoặc -trake.csv.`);
        }

        const rows = parseCsv(text);
        if (!rows.length) throw new Error(`${fileName}: file CSV trống.`);
        if (rows.length > 100) throw new Error(`${fileName}: vượt quá 100 dòng.`);

        let eventCount = 0;
        const pinned = rows.map((cells, index) => {
            const rowNumber = index + 1;
            const videoId = String(cells[0] || '').trim().toUpperCase();
            if (!/^L\d{2}_V\d+$/.test(videoId)) {
                throw new Error(`${fileName}, dòng ${rowNumber}: video_id "${videoId}" không hợp lệ.`);
            }

            if (type === 'kis') {
                if (cells.length !== 2) {
                    throw new Error(`${fileName}, dòng ${rowNumber}: KIS cần đúng 2 cột.`);
                }
                return {
                    videoId,
                    frameIdx: parseFrameIndex(cells[1], fileName, rowNumber),
                    path: ''
                };
            }

            if (type === 'qa') {
                if (cells.length !== 3) {
                    throw new Error(`${fileName}, dòng ${rowNumber}: QA cần đúng 3 cột.`);
                }
                const answer = String(cells[2] || '').trim();
                if (!answer) throw new Error(`${fileName}, dòng ${rowNumber}: thiếu answer.`);
                if (answer.length > 100) {
                    throw new Error(`${fileName}, dòng ${rowNumber}: answer vượt quá 100 ký tự.`);
                }
                return {
                    videoId,
                    frameIdx: parseFrameIndex(cells[1], fileName, rowNumber),
                    answer,
                    path: ''
                };
            }

            const currentEventCount = cells.length - 1;
            if (currentEventCount < 2) {
                throw new Error(`${fileName}, dòng ${rowNumber}: TRAKE cần ít nhất 2 frame_idx.`);
            }
            if (!eventCount) eventCount = currentEventCount;
            if (currentEventCount !== eventCount) {
                throw new Error(`${fileName}, dòng ${rowNumber}: số event không đồng nhất (cần ${eventCount}).`);
            }
            const frameIndices = cells.slice(1)
                .map(value => parseFrameIndex(value, fileName, rowNumber));
            if (frameIndices.some((value, position) => position > 0 && value <= frameIndices[position - 1])) {
                throw new Error(`${fileName}, dòng ${rowNumber}: frame TRAKE phải tăng theo thời gian.`);
            }
            return { videoId, frameIndices, paths: [] };
        });

        const firstAnswer = type === 'qa' ? String(pinned[0]?.answer || '') : '';
        return {
            id,
            type,
            eventCount: type === 'trake' ? eventCount : 0,
            prompt: '',
            answer: firstAnswer,
            pinned,
            pool: [],
            automatic: []
        };
    }

    async function importSubmissionFolder() {
        const csvFiles = Array.from(elements.importSubmissionFolder.files || [])
            .filter(file => file.name.toLowerCase().endsWith('.csv'))
            .sort((left, right) => {
                const leftPath = left.webkitRelativePath || left.name;
                const rightPath = right.webkitRelativePath || right.name;
                return leftPath.localeCompare(rightPath, undefined, { numeric: true });
            });
        if (!csvFiles.length) throw new Error('Folder không có file .csv nào.');

        const incoming = { version: 1, activeQueryId: '', queries: {}, latestPools: {} };
        for (const file of csvFiles) {
            const imported = parseSubmissionCsv(file.name, await file.text());
            const duplicateId = Object.keys(incoming.queries)
                .find(id => id.toLowerCase() === imported.id.toLowerCase());
            if (!duplicateId) {
                incoming.queries[imported.id] = imported;
                if (!incoming.activeQueryId) incoming.activeQueryId = imported.id;
                continue;
            }

            const existing = incoming.queries[duplicateId];
            if (existing.type !== imported.type
                    || (existing.type === 'trake' && existing.eventCount !== imported.eventCount)) {
                throw new Error(`Các file trùng tên query "${imported.id}" nhưng khác cấu trúc.`);
            }
            existing.pinned.push(...imported.pinned);
            if (!existing.answer && imported.answer) existing.answer = imported.answer;
        }

        const current = PrelimSubmission.load();
        const merged = JSON.parse(JSON.stringify(current));
        const summary = mergeProjectStates(merged, incoming);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        downloadJson(`aic26-before-csv-folder-merge-${timestamp}.json`, current);
        PrelimSubmission.save(merged);
        message(
            `Đã merge ${csvFiles.length} CSV: ${summary.newQueries} query mới, `
            + `${summary.mergedQueries} query cũ được gộp, ${summary.addedPinned} dòng mới. `
            + `Bỏ ${summary.duplicates} dòng trùng và ${summary.overflow} dòng vượt giới hạn.`
        );
    }

    function normalizeImportedCandidate(type, rawCandidate, queryId, eventCount, allowUnresolved = false) {
        if (!rawCandidate || typeof rawCandidate !== 'object' || Array.isArray(rawCandidate)) {
            throw new Error(`Query "${queryId}" chứa candidate không hợp lệ.`);
        }
        const videoId = String(rawCandidate.videoId || rawCandidate.video_id || '').trim().toUpperCase();
        if (!/^L\d{2}_V\d+$/.test(videoId)) {
            throw new Error(`Query "${queryId}" có video_id "${videoId}" không hợp lệ.`);
        }
        const path = String(rawCandidate.path || rawCandidate.web_path || '');
        const score = Number(rawCandidate.score || 0);
        const rawPtsTime = rawCandidate.ptsTime ?? rawCandidate.pts_time;
        const ptsTime = rawPtsTime === undefined || rawPtsTime === null || rawPtsTime === ''
            ? undefined
            : Number(rawPtsTime);
        if (ptsTime !== undefined && (!Number.isFinite(ptsTime) || ptsTime < 0)) {
            throw new Error(`Query "${queryId}" có timestamp không hợp lệ.`);
        }

        if (type === 'trake') {
            const rawFrames = rawCandidate.frameIndices || rawCandidate.frame_indices;
            if (!Array.isArray(rawFrames)) {
                throw new Error(`Query "${queryId}" có dòng TRAKE thiếu frameIndices.`);
            }
            const frameIndices = rawFrames.map(Number);
            if (frameIndices.some(value => !Number.isSafeInteger(value) || value < 0)) {
                throw new Error(`Query "${queryId}" có frame TRAKE không hợp lệ.`);
            }
            if (frameIndices.length !== eventCount) {
                throw new Error(`Query "${queryId}" cần đúng ${eventCount} frame cho mỗi dòng TRAKE.`);
            }
            if (frameIndices.some((value, index) => index > 0 && value <= frameIndices[index - 1])) {
                throw new Error(`Query "${queryId}" có frame TRAKE không tăng theo thời gian.`);
            }
            return {
                videoId,
                frameIndices,
                paths: Array.isArray(rawCandidate.paths) ? rawCandidate.paths.map(value => String(value || '')) : [],
                score: Number.isFinite(score) ? score : 0,
            };
        }

        const rawFrameIdx = rawCandidate.frameIdx ?? rawCandidate.frame_idx;
        if (rawFrameIdx === undefined || rawFrameIdx === null || rawFrameIdx === '') {
            if (allowUnresolved && path) {
                return {
                    videoId,
                    path,
                    frame_n: rawCandidate.frame_n,
                    score: Number.isFinite(score) ? score : 0,
                };
            }
            throw new Error(`Query "${queryId}" có candidate thiếu frame_idx.`);
        }
        const frameIdx = Number(rawFrameIdx);
        if (!Number.isSafeInteger(frameIdx) || frameIdx < 0) {
            throw new Error(`Query "${queryId}" có frame_idx "${rawFrameIdx}" không hợp lệ.`);
        }
        const answer = String(rawCandidate.answer || '');
        if (answer.length > 100) {
            throw new Error(`Query "${queryId}" có answer vượt quá 100 ký tự.`);
        }
        return {
            videoId,
            frameIdx,
            path,
            ...(ptsTime !== undefined ? { ptsTime } : {}),
            ...(type === 'qa' && answer ? { answer } : {}),
            score: Number.isFinite(score) ? score : 0,
        };
    }

    function normalizedImportedQuery(queryId, rawQuery) {
        const id = normalizeQueryId(queryId);
        const type = validatedQueryType(id);
        if (!type || !rawQuery || typeof rawQuery !== 'object' || Array.isArray(rawQuery)) {
            throw new Error(`Query "${queryId}" không hợp lệ.`);
        }
        const declaredType = String(rawQuery.type || type).toLowerCase();
        if (declaredType !== type) {
            throw new Error(`Query "${id}" có type ${declaredType}, không khớp hậu tố -${type}.`);
        }
        const candidateLists = ['pinned', 'automatic', 'pool']
            .flatMap(field => Array.isArray(rawQuery[field]) ? rawQuery[field] : []);
        const trakeSample = type === 'trake'
            ? candidateLists.find(candidate => Array.isArray(candidate?.frameIndices || candidate?.frame_indices))
            : null;
        const inferredEventCount = trakeSample
            ? (trakeSample.frameIndices || trakeSample.frame_indices).length
            : (type === 'trake' ? 2 : 0);
        const rawEventCount = Number(rawQuery.eventCount || inferredEventCount || 2);
        const eventCount = type === 'trake' ? normalizeEventCount(rawEventCount) : 0;
        const answer = String(rawQuery.answer || '');
        if (answer.length > 100) throw new Error(`Query "${id}" có answer vượt quá 100 ký tự.`);
        const normalizeList = (field, allowUnresolved = false) => (
            Array.isArray(rawQuery[field])
                ? rawQuery[field].map(candidate => normalizeImportedCandidate(
                    type, candidate, id, eventCount, allowUnresolved
                ))
                : []
        );
        return {
            ...rawQuery,
            id,
            type,
            eventCount,
            prompt: String(rawQuery.prompt || ''),
            answer,
            neighborTimeBorder: Math.max(1, Math.min(600, Number(rawQuery.neighborTimeBorder) || 30)),
            pinned: normalizeList('pinned'),
            pool: normalizeList('pool', true),
            automatic: normalizeList('automatic'),
        };
    }

    function mergeCandidates(type, localItems, remoteItems, limit, summary, countPinned = false) {
        const output = [];
        const seen = new Set();
        for (const candidate of [...(localItems || []), ...(remoteItems || [])]) {
            if (!candidate || typeof candidate !== 'object') {
                summary.duplicates += 1;
                continue;
            }
            const key = PrelimSubmission.candidateKey(type, candidate);
            if (seen.has(key)) {
                summary.duplicates += 1;
                continue;
            }
            seen.add(key);
            if (output.length < limit) {
                output.push(candidate);
                if (countPinned && (remoteItems || []).includes(candidate)) summary.addedPinned += 1;
            } else {
                summary.overflow += 1;
            }
        }
        return output;
    }

    function mergeProjectStates(localState, incomingState) {
        if (!incomingState || incomingState.version !== 1 || !incomingState.queries
                || typeof incomingState.queries !== 'object' || Array.isArray(incomingState.queries)) {
            throw new Error('Project merge không đúng schema version 1.');
        }
        if (!localState || !localState.queries || typeof localState.queries !== 'object') {
            throw new Error('Project local đang hỏng schema; hãy khôi phục từ backup JSON.');
        }
        const summary = {
            newQueries: 0,
            mergedQueries: 0,
            addedPinned: 0,
            duplicates: 0,
            conflicts: 0,
            overflow: 0,
        };
        const incomingQueries = Object.entries(incomingState.queries || {});

        for (const [incomingId, rawQuery] of incomingQueries) {
            const remote = normalizedImportedQuery(incomingId, rawQuery);
            const localId = Object.keys(localState.queries)
                .find(id => id.toLowerCase() === remote.id.toLowerCase());

            if (!localId) {
                const pinned = mergeCandidates(remote.type, [], remote.pinned, 100, summary, true);
                const pinnedKeys = new Set(pinned.map(item => PrelimSubmission.candidateKey(remote.type, item)));
                const automatic = mergeCandidates(
                    remote.type,
                    [],
                    remote.automatic.filter(item => item && typeof item === 'object'
                        && !pinnedKeys.has(PrelimSubmission.candidateKey(remote.type, item))),
                    Math.max(0, 100 - pinned.length),
                    summary
                );
                localState.queries[remote.id] = {
                    ...remote,
                    pinned,
                    pool: mergeCandidates(remote.type, [], remote.pool, 1000, summary),
                    automatic,
                };
                summary.newQueries += 1;
                continue;
            }

            const local = normalizedImportedQuery(localId, localState.queries[localId]);
            if (local.type !== remote.type) {
                throw new Error(`Query "${localId}" bị trùng tên nhưng khác loại.`);
            }
            summary.mergedQueries += 1;

            local.pinned = mergeCandidates(local.type, local.pinned, remote.pinned, 100, summary, true);
            const pinnedKeys = new Set(local.pinned.map(item => PrelimSubmission.candidateKey(local.type, item)));
            const automaticCandidates = [...local.automatic, ...remote.automatic]
                .filter(item => item && typeof item === 'object'
                    && !pinnedKeys.has(PrelimSubmission.candidateKey(local.type, item)));
            local.automatic = mergeCandidates(
                local.type,
                [],
                automaticCandidates,
                Math.max(0, 100 - local.pinned.length),
                summary
            );
            local.pool = mergeCandidates(local.type, local.pool, remote.pool, 1000, summary);

            if (!local.prompt && remote.prompt) local.prompt = remote.prompt;
            else if (local.prompt && remote.prompt && local.prompt !== remote.prompt) summary.conflicts += 1;
            if (!local.answer && remote.answer) local.answer = remote.answer;
            else if (local.answer && remote.answer && local.answer !== remote.answer) summary.conflicts += 1;
            if (local.type === 'trake' && local.eventCount !== remote.eventCount) summary.conflicts += 1;

            localState.queries[localId] = local;
        }

        const localPools = localState.latestPools || {};
        const remotePools = incomingState.latestPools || {};
        localState.latestPools = {
            frame: mergeCandidates('kis', localPools.frame, remotePools.frame, 1000, summary),
            trake: mergeCandidates('trake', localPools.trake, remotePools.trake, 1000, summary),
            trakeEventCount: Number(localPools.trakeEventCount || remotePools.trakeEventCount || 0),
        };
        if (!localState.activeQueryId || !localState.queries[localState.activeQueryId]) {
            const requested = String(incomingState.activeQueryId || '').toLowerCase();
            localState.activeQueryId = Object.keys(localState.queries)
                .find(id => id.toLowerCase() === requested) || Object.keys(localState.queries)[0] || '';
        }
        localState.version = 1;
        return summary;
    }

    elements.newQueryId.addEventListener('input', () => {
        elements.eventCountRow.classList.toggle(
            'hidden', PrelimSubmission.queryTypeFromId(elements.newQueryId.value) !== 'trake'
        );
    });
    elements.addQuery.addEventListener('click', () => {
        try {
            createQuery(elements.newQueryId.value);
            elements.newQueryId.value = '';
            elements.eventCountRow.classList.add('hidden');
            message('Đã tạo query. Hãy mở trang search và chọn query này để ghim kết quả.');
        } catch (error) {
            message(error.message, true);
        }
    });
    elements.newQueryId.addEventListener('keydown', event => {
        if (event.key === 'Enter') elements.addQuery.click();
    });
    elements.importQueryFiles.addEventListener('change', async () => {
        try {
            for (const file of elements.importQueryFiles.files) {
                createQuery(file.name, await file.text());
            }
            message(`Đã import ${elements.importQueryFiles.files.length} file query.`);
        } catch (error) {
            message(error.message, true);
        } finally {
            elements.importQueryFiles.value = '';
        }
    });
    elements.queryPrompt.addEventListener('input', () => saveQueryField('prompt', elements.queryPrompt.value));
    elements.qaAnswer.addEventListener('input', () => {
        saveQueryField('answer', elements.qaAnswer.value);
        elements.answerLength.textContent = `${elements.qaAnswer.value.length}/100`;
    });
    elements.activeEventCount.addEventListener('change', () => {
        saveQueryField('eventCount', normalizeEventCount(elements.activeEventCount.value));
    });
    elements.addManualCandidate.addEventListener('click', addManualCandidate);
    elements.manualCandidate.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
            event.preventDefault();
            addManualCandidate();
        }
    });
    elements.fillResults.addEventListener('click', fillFromPinned);
    elements.neighborTimeBorder.addEventListener('change', () => {
        const value = Math.max(1, Math.min(600, Number(elements.neighborTimeBorder.value) || 30));
        elements.neighborTimeBorder.value = value;
        saveQueryField('neighborTimeBorder', value);
    });
    elements.clearAutomatic.addEventListener('click', () => {
        saveQueryField('automatic', []);
        message('Đã xóa phần auto-fill; các dòng ghim vẫn được giữ nguyên.');
    });
    elements.previewCsv.addEventListener('click', () => {
        const query = activeQuery();
        if (!query) return;
        elements.csvPreview.value = csvPreviewFor(query);
        elements.csvPreviewPanel.classList.remove('hidden');
    });
    elements.closePreview.addEventListener('click', () => elements.csvPreviewPanel.classList.add('hidden'));
    elements.deleteQuery.addEventListener('click', () => {
        const state = PrelimSubmission.load();
        const query = activeQuery(state);
        if (!query || !confirm(`Xóa query ${query.id} và toàn bộ kết quả đã ghim?`)) return;
        delete state.queries[query.id];
        state.activeQueryId = Object.keys(state.queries)[0] || '';
        PrelimSubmission.save(state);
        message('Đã xóa query.');
    });
    elements.exportZip.addEventListener('click', exportZip);
    elements.exportProject.addEventListener('click', () => {
        downloadJson('aic26-submission-project.json', PrelimSubmission.load());
    });
    elements.importProject.addEventListener('change', async () => {
        try {
            const file = elements.importProject.files[0];
            if (!file) return;
            const parsed = JSON.parse(await file.text());
            if (!parsed || parsed.version !== 1 || !parsed.queries
                    || typeof parsed.queries !== 'object' || Array.isArray(parsed.queries)) {
                throw new Error('File project không đúng schema version 1.');
            }
            const current = PrelimSubmission.load();
            const merged = JSON.parse(JSON.stringify(current));
            const summary = mergeProjectStates(merged, parsed);
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            downloadJson(`aic26-before-merge-${timestamp}.json`, current);
            PrelimSubmission.save(merged);
            message(
                `Merge xong: ${summary.newQueries} query mới, ${summary.mergedQueries} query trùng được gộp, `
                + `${summary.addedPinned} dòng ghim mới. Bỏ ${summary.duplicates} dòng trùng, `
                + `giữ local ở ${summary.conflicts} conflict, bỏ ${summary.overflow} dòng vượt giới hạn.`
            );
        } catch (error) {
            message(`Merge thất bại, project hiện tại không bị thay đổi: ${error.message}`, true);
        } finally {
            elements.importProject.value = '';
        }
    });
    elements.importSubmissionFolder.addEventListener('change', async () => {
        try {
            await importSubmissionFolder();
        } catch (error) {
            message(`Merge folder CSV thất bại, project hiện tại không bị thay đổi: ${error.message}`, true);
        } finally {
            elements.importSubmissionFolder.value = '';
        }
    });
    elements.closeFrameDetail.addEventListener('click', closeFrameDetail);
    elements.frameDetailModal.addEventListener('click', event => {
        if (event.target === elements.frameDetailModal) closeFrameDetail();
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && !elements.frameDetailModal.classList.contains('hidden')) {
            closeFrameDetail();
        }
    });

    window.addEventListener('storage', render);
    window.addEventListener('prelim-submission-changed', render);
    render();
})();
