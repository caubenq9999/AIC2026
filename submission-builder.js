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
                ? Math.max(
                    2,
                    Number(state.latestPools?.trakeEventCount || elements.newEventCount.value || 2)
                )
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
                thumbnail.title = 'Bấm để xem video';
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
            if (thumbnail) thumbnail.addEventListener('click', () => watchButton.click());
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
        saveQueryField('eventCount', Math.max(2, Number(elements.activeEventCount.value || 2)));
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
            const parsed = JSON.parse(await elements.importProject.files[0].text());
            if (!parsed || parsed.version !== 1 || typeof parsed.queries !== 'object') {
                throw new Error('File project không đúng schema version 1.');
            }
            PrelimSubmission.save(parsed);
            message('Đã khôi phục project JSON.');
        } catch (error) {
            message(error.message, true);
        } finally {
            elements.importProject.value = '';
        }
    });

    window.addEventListener('storage', render);
    window.addEventListener('prelim-submission-changed', render);
    render();
})();
