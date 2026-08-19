(function () {
    'use strict';

    const STORAGE_KEY = 'aic26-preliminary-submission-v1';

    function emptyState() {
        return {
            version: 1,
            activeQueryId: '',
            queries: {},
            latestPools: { frame: [], trake: [], trakeEventCount: 0 },
        };
    }

    function load() {
        try {
            const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
            if (!parsed || parsed.version !== 1 || typeof parsed.queries !== 'object') {
                return emptyState();
            }
            parsed.activeQueryId = String(parsed.activeQueryId || '');
            if (!parsed.latestPools || typeof parsed.latestPools !== 'object') {
                parsed.latestPools = { frame: [], trake: [], trakeEventCount: 0 };
            }
            parsed.latestPools.frame = Array.isArray(parsed.latestPools.frame)
                ? parsed.latestPools.frame
                : [];
            parsed.latestPools.trake = Array.isArray(parsed.latestPools.trake)
                ? parsed.latestPools.trake
                : [];
            parsed.latestPools.trakeEventCount = Number(parsed.latestPools.trakeEventCount || 0);
            return parsed;
        } catch (error) {
            console.warn('Không đọc được bản nháp submission:', error);
            return emptyState();
        }
    }

    function save(state) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        window.dispatchEvent(new CustomEvent('prelim-submission-changed'));
    }

    function queryTypeFromId(queryId) {
        const match = String(queryId || '').trim().match(/-(kis|qa|trake)(?:\.(?:txt|csv))?$/i);
        return match ? match[1].toLowerCase() : '';
    }

    function candidateKey(type, candidate) {
        const videoId = String(candidate.videoId || '').toUpperCase();
        if (type === 'trake') {
            return `${videoId}|${(candidate.frameIndices || []).join(',')}`;
        }
        const framePart = candidate.frameIdx !== undefined && candidate.frameIdx !== null
            ? candidate.frameIdx
            : candidate.path || '';
        return `${videoId}|${framePart}`;
    }

    function dedupe(type, candidates) {
        const seen = new Set();
        const output = [];
        for (const candidate of candidates || []) {
            const key = candidateKey(type, candidate);
            if (!key || seen.has(key)) continue;
            seen.add(key);
            output.push(candidate);
        }
        return output;
    }

    function finalRows(query) {
        const type = query.type;
        const pinned = dedupe(type, query.pinned || []);
        const pinnedKeys = new Set(pinned.map(item => candidateKey(type, item)));
        const automatic = dedupe(type, query.automatic || [])
            .filter(item => !pinnedKeys.has(candidateKey(type, item)));
        return [...pinned, ...automatic].slice(0, 100);
    }

    function totalPinned(state) {
        return Object.values(state.queries || {})
            .reduce((sum, query) => sum + (query.pinned || []).length, 0);
    }

    window.PrelimSubmission = {
        STORAGE_KEY,
        load,
        save,
        emptyState,
        queryTypeFromId,
        candidateKey,
        dedupe,
        finalRows,
        totalPinned,
    };
})();
