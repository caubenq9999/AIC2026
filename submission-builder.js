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
        frameDetailModal: document.getElementById('frame-detail-modal'),
        frameDetailQuery: document.getElementById('frame-detail-query'),
        frameDetailTitle: document.getElementById('frame-detail-title'),
        frameDetailEvents: document.getElementById('frame-detail-events'),
        frameDetailLoading: document.getElementById('frame-detail-loading'),
        frameDetailContent: document.getElementById('frame-detail-content'),
        frameDetailPlayer: document.getElementById('frame-detail-player'),
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

    function normalizeEventCount(value) {
        const count = Number(value);
        return Number.isFinite(count) ? Math.max(2, Math.min(30, Math.trunc(count))) : 2;
    }

    function createQuery(queryId, prompt = '') {
        const id = normalizeQueryId(queryId);
        const type = PrelimSubmission.queryTypeFromId(id);
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
            return `${base},${query.answer ? csvCell(query.answer) : '<thiếu answer>'}`;
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
            detail.textContent = `${query.type.toUpperCase()} · pool ${(query.pool || []).length}`;
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
            const endpoint = keyframePath ? '/metadata' : '/submission/playback';
            const requestBody = keyframePath
                ? { image_path: keyframePath }
                : { videoId: candidate.videoId, frameIdx };
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(data.error || 'Không tìm thấy video.');
            const target = data.playback_url || data.path;
            if (!target) throw new Error('Video này không có playback URL hoặc keyframe fallback.');
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
    }

    async function loadFrameDetail(query, candidate, eventIndex = 0) {
        const isTrake = query.type === 'trake';
        const frameIdx = isTrake ? (candidate.frameIndices || [])[eventIndex] : candidate.frameIdx;
        const keyframePath = isTrake ? (candidate.paths || [])[eventIndex] : candidate.path;
        elements.frameDetailLoading.textContent = 'Đang tải metadata...';
        elements.frameDetailLoading.classList.remove('hidden');
        elements.frameDetailContent.classList.add('hidden');
        elements.frameDetailPlayer.src = '';

        Array.from(elements.frameDetailEvents.children).forEach((button, index) => {
            button.classList.toggle('active', index === eventIndex);
        });

        try {
            let response;
            if (keyframePath) {
                response = await fetch(`${API_BASE_URL}/metadata`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_path: keyframePath }),
                });
            } else {
                response = await fetch(`${API_BASE_URL}/submission/playback`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ videoId: candidate.videoId, frameIdx }),
                });
            }
            const meta = await response.json();
            if (!response.ok || meta.error) throw new Error(meta.error || 'Không đọc được metadata.');

            const imagePath = keyframePath || meta.path || '';
            const playbackUrl = meta.playback_url || '';
            const ptsTime = Number(meta.pts_time || 0);
            const embedUrl = youtubeEmbedUrl(playbackUrl, ptsTime);
            elements.frameDetailTitle.textContent = candidate.videoId;
            elements.frameDetailVideoId.textContent = candidate.videoId;
            elements.frameDetailFrameN.textContent = meta.n ?? 'N/A';
            elements.frameDetailFrameIdx.textContent = frameIdx ?? meta.frameIdx ?? meta.frame_idx ?? 'N/A';
            elements.frameDetailTime.textContent = `${ptsTime.toFixed(2)} giây`;
            elements.frameDetailImage.src = imagePath
                ? new URL(imagePath, `${API_BASE_URL}/`).href
                : '';
            elements.frameDetailImage.classList.toggle('hidden', !imagePath);
            elements.frameDetailPlayer.src = embedUrl;
            elements.frameDetailVideoMissing.classList.toggle('hidden', Boolean(embedUrl));
            elements.frameDetailOpenVideo.href = playbackUrl || '#';
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
            if (isPinned) tr.className = 'pinned-row';

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
                actionsCell.append(
                    actionButton('↑', 'Đưa lên', () => mutatePinned(index, 'up')),
                    actionButton('↓', 'Đưa xuống', () => mutatePinned(index, 'down')),
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
            `${(query.pinned || []).length} ghim · ${(query.pool || []).length} trong ranking gần nhất · ${(query.automatic || []).length} auto-fill`;
        elements.queryPrompt.value = query.prompt || '';
        elements.qaAnswerRow.classList.toggle('hidden', query.type !== 'qa');
        elements.qaAnswer.value = query.answer || '';
        elements.answerLength.textContent = `${(query.answer || '').length}/100`;
        elements.trakeEventRow.classList.toggle('hidden', query.type !== 'trake');
        elements.activeEventCount.value = query.eventCount || 2;
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

    async function fillFromPool() {
        const state = PrelimSubmission.load();
        const query = activeQuery(state);
        if (!query) return;
        if (!(query.pool || []).length) {
            message(
                'Query này chưa có ranking. Quay lại trang search, chọn đúng query và chạy search một lần; sau đó ranking sẽ tự lưu ở đây.',
                true
            );
            return;
        }
        if (query.type !== 'trake' && query.pool.some(item => item.frameIdx === undefined || item.frameIdx === null)) {
            const oldText = elements.fillResults.textContent;
            elements.fillResults.disabled = true;
            elements.fillResults.textContent = 'Đang map frame_idx...';
            try {
                const response = await fetch(`${API_BASE_URL}/submission/resolve_candidates`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ candidates: query.pool }),
                });
                const data = await response.json();
                if (!response.ok || data.error) throw new Error(data.error || 'Không map được ranking.');
                query.pool = data.resolved || [];
                if (!query.pool.length) throw new Error('Không có frame nào map được sang frame_idx.');
            } catch (error) {
                message(`Không fill được ranking: ${error.message}`, true);
                return;
            } finally {
                elements.fillResults.disabled = false;
                elements.fillResults.textContent = oldText;
            }
        }
        const pinned = PrelimSubmission.dedupe(query.type, query.pinned || []);
        const pinnedKeys = new Set(pinned.map(item => PrelimSubmission.candidateKey(query.type, item)));
        const needed = Math.max(0, 100 - pinned.length);
        query.automatic = PrelimSubmission.dedupe(query.type, query.pool || [])
            .filter(item => !pinnedKeys.has(PrelimSubmission.candidateKey(query.type, item)))
            .slice(0, needed);
        PrelimSubmission.save(state);
        message(`Đã giữ ${pinned.length} dòng thủ công và fill ${query.automatic.length} dòng từ ranking.`);
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

    function normalizedImportedQuery(queryId, rawQuery) {
        const id = normalizeQueryId(queryId);
        const type = PrelimSubmission.queryTypeFromId(id);
        if (!type || !rawQuery || typeof rawQuery !== 'object' || Array.isArray(rawQuery)) {
            throw new Error(`Query "${queryId}" không hợp lệ.`);
        }
        const declaredType = String(rawQuery.type || type).toLowerCase();
        if (declaredType !== type) {
            throw new Error(`Query "${id}" có type ${declaredType}, không khớp hậu tố -${type}.`);
        }
        const rawEventCount = Number(rawQuery.eventCount || 2);
        return {
            ...rawQuery,
            id,
            type,
            eventCount: type === 'trake' && Number.isFinite(rawEventCount)
                ? Math.max(2, rawEventCount)
                : (type === 'trake' ? 2 : 0),
            prompt: String(rawQuery.prompt || ''),
            answer: String(rawQuery.answer || ''),
            pinned: Array.isArray(rawQuery.pinned) ? rawQuery.pinned : [],
            pool: Array.isArray(rawQuery.pool) ? rawQuery.pool : [],
            automatic: Array.isArray(rawQuery.automatic) ? rawQuery.automatic : [],
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
    elements.fillResults.addEventListener('click', fillFromPool);
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
