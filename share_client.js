(() => {
    "use strict";

    const POLL_INTERVAL_MS = 1500;
    const STORAGE = {
        server: "aic-share-server",
        name: "aic-share-display-name",
    };

    let serverBaseUrl = "";
    let displayName = "";
    let lastMessageId = 0;
    let attachedVideo = null;
    let panelOpen = false;
    let unreadCount = 0;
    let polling = false;
    let pollCount = 0;
    const renderedMessageIds = new Set();

    const byId = (id) => document.getElementById(id);

    function normalizeServerUrl(value) {
        let normalized = String(value || "").trim();
        if (!normalized) {
            normalized = `http://${window.location.hostname || "localhost"}:5050`;
        }
        if (!/^https?:\/\//i.test(normalized)) normalized = `http://${normalized}`;
        return normalized.replace(/\/+$/, "");
    }

    function defaultName() {
        const suffix = Math.floor(100 + Math.random() * 900);
        return `Player-${suffix}`;
    }

    function setConnectionState(online, label) {
        byId("chat-connection-dot").classList.toggle("online", online);
        byId("chat-connection-label").textContent = label || (online ? "Đã kết nối" : "Mất kết nối");
    }

    async function fetchWithTimeout(url, options = {}) {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), 4000);
        try {
            return await fetch(url, { ...options, signal: controller.signal });
        } finally {
            window.clearTimeout(timeout);
        }
    }

    function updateUnreadBadge() {
        const badge = byId("chat-unread-badge");
        badge.textContent = unreadCount > 99 ? "99+" : String(unreadCount);
        badge.classList.toggle("hidden", unreadCount === 0);
    }

    function togglePanel(forceOpen) {
        panelOpen = typeof forceOpen === "boolean" ? forceOpen : !panelOpen;
        byId("team-chat-popup").classList.toggle("hidden", !panelOpen);
        byId("team-chat-toggle").setAttribute("aria-expanded", String(panelOpen));
        if (panelOpen) {
            unreadCount = 0;
            updateUnreadBadge();
            byId("chat-message-input").focus();
            scrollMessagesToBottom();
            pollMessages();
        }
    }

    function scrollMessagesToBottom() {
        const list = byId("chat-message-list");
        window.requestAnimationFrame(() => {
            list.scrollTop = list.scrollHeight;
        });
    }

    function formatTime(value) {
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return "";
        return new Intl.DateTimeFormat("vi-VN", {
            hour: "2-digit",
            minute: "2-digit",
        }).format(date);
    }

    function avatarText(name) {
        return name.trim().split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "?";
    }

    function localImagePath(value) {
        if (!value) return "";
        try {
            const url = new URL(value, window.location.origin);
            return `${url.pathname}${url.search}`;
        } catch (_) {
            return value;
        }
    }

    function findGalleryImage(imagePath) {
        const target = localImagePath(imagePath);
        return [...document.querySelectorAll(".gallery-item")].find((image) => {
            return localImagePath(image.getAttribute("src") || image.src) === target;
        }) || null;
    }

    function openSharedVideo(video) {
        if (!video || !video.image_path || typeof window.showImageDetail !== "function") return;
        window.showImageDetail(video.image_path, findGalleryImage(video.image_path));
        togglePanel(false);
    }

    function createVideoCard(video) {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "chat-video-card";
        card.title = "Mở frame này trong VR";

        if (video.image_path) {
            const image = document.createElement("img");
            image.src = video.image_path;
            image.alt = `Keyframe ${video.video_id}`;
            image.loading = "lazy";
            image.addEventListener("error", () => image.remove());
            card.appendChild(image);
        }

        const info = document.createElement("span");
        info.className = "chat-video-info";
        const title = document.createElement("strong");
        title.textContent = video.video_id;
        const meta = document.createElement("span");
        const details = [];
        if (video.pts_time !== null && video.pts_time !== undefined) {
            details.push(`${Number(video.pts_time).toFixed(2)}s`);
        }
        if (video.frame_n !== null && video.frame_n !== undefined) details.push(`frame ${video.frame_n}`);
        meta.textContent = details.join(" · ") || "Kết quả retrieval";
        info.append(title, meta);
        if (video.query) {
            const query = document.createElement("span");
            query.className = "chat-video-query";
            query.textContent = video.query;
            info.appendChild(query);
        }
        card.appendChild(info);
        card.addEventListener("click", () => openSharedVideo(video));
        return card;
    }

    function appendMessage(message, countAsUnread = true) {
        const numericId = Number(message.id) || 0;
        if (numericId && renderedMessageIds.has(numericId)) return;
        if (numericId) {
            renderedMessageIds.add(numericId);
            lastMessageId = Math.max(lastMessageId, numericId);
        }

        const empty = byId("chat-empty-state");
        if (empty) empty.classList.add("hidden");

        const own = message.sender === displayName;
        const row = document.createElement("div");
        row.className = `chat-message-row${own ? " own" : ""}`;
        row.dataset.messageId = String(numericId);

        if (!own) {
            const avatar = document.createElement("span");
            avatar.className = "chat-avatar";
            avatar.textContent = avatarText(message.sender);
            row.appendChild(avatar);
        }

        const body = document.createElement("div");
        body.className = "chat-message-body";
        if (!own) {
            const sender = document.createElement("span");
            sender.className = "chat-message-sender";
            sender.textContent = message.sender;
            body.appendChild(sender);
        }

        const bubble = document.createElement("div");
        bubble.className = "chat-message-bubble";
        if (message.text) {
            const text = document.createElement("p");
            text.textContent = message.text;
            bubble.appendChild(text);
        }
        if (message.video) bubble.appendChild(createVideoCard(message.video));

        const time = document.createElement("span");
        time.className = "chat-message-time";
        time.textContent = formatTime(message.created_at);
        const footer = document.createElement("span");
        footer.className = "chat-message-footer";
        footer.appendChild(time);
        if (own && numericId) {
            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "chat-delete-message";
            remove.setAttribute("aria-label", "Xóa tin nhắn");
            remove.title = "Xóa tin nhắn";
            remove.textContent = "×";
            remove.addEventListener("click", () => deleteMessage(numericId, row));
            footer.appendChild(remove);
        }
        body.append(bubble, footer);
        row.appendChild(body);
        byId("chat-message-list").appendChild(row);

        if (!panelOpen && !own && countAsUnread) {
            unreadCount += 1;
            updateUnreadBadge();
        }
        scrollMessagesToBottom();
    }

    function showEmptyStateIfNeeded() {
        const hasMessages = Boolean(byId("chat-message-list").querySelector(".chat-message-row"));
        byId("chat-empty-state").classList.toggle("hidden", hasMessages);
    }

    function reconcileMessages(messages) {
        const serverIds = new Set(messages.map((message) => Number(message.id)).filter(Boolean));
        byId("chat-message-list").querySelectorAll(".chat-message-row[data-message-id]").forEach((row) => {
            const id = Number(row.dataset.messageId);
            if (!serverIds.has(id)) {
                row.remove();
                renderedMessageIds.delete(id);
            }
        });
        showEmptyStateIfNeeded();
    }

    async function pollMessages(initial = false) {
        if (polling) return;
        polling = true;
        try {
            pollCount += 1;
            const fullSync = initial || pollCount % 10 === 0;
            const after = fullSync ? 0 : lastMessageId;
            const response = await fetchWithTimeout(`${serverBaseUrl}/api/messages?after=${after}&limit=100`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (fullSync) reconcileMessages(data.messages || []);
            (data.messages || []).forEach((message) => appendMessage(message, !initial));
            setConnectionState(true, "Đã kết nối");
        } catch (error) {
            setConnectionState(false, "Host offline");
        } finally {
            polling = false;
        }
    }

    async function deleteMessage(messageId, row) {
        try {
            const response = await fetchWithTimeout(`${serverBaseUrl}/api/messages/${messageId}`, {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sender: displayName }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            row.remove();
            renderedMessageIds.delete(messageId);
            showEmptyStateIfNeeded();
        } catch (error) {
            byId("chat-composer-hint").textContent = error.message || "Không xóa được tin nhắn.";
        }
    }

    async function clearOwnMessages() {
        if (!window.confirm("Xóa toàn bộ tin nhắn bạn đã gửi trên máy host?")) return;
        try {
            const response = await fetchWithTimeout(`${serverBaseUrl}/api/messages`, {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sender: displayName }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            byId("chat-message-list").querySelectorAll(".chat-message-row.own").forEach((row) => {
                renderedMessageIds.delete(Number(row.dataset.messageId));
                row.remove();
            });
            showEmptyStateIfNeeded();
            byId("chat-composer-hint").textContent = `Đã xóa ${data.deleted || 0} tin nhắn của bạn.`;
        } catch (error) {
            byId("chat-composer-hint").textContent = error.message || "Không dọn được tin nhắn.";
        }
    }

    function readCurrentVideo() {
        const detail = byId("image-detail");
        const videoId = byId("video-name").textContent.trim();
        if (detail.classList.contains("hidden") || !videoId || videoId === "N/A") return null;

        const parseNumber = (id, integer = false) => {
            const value = byId(id).textContent.trim();
            const parsed = integer ? Number.parseInt(value, 10) : Number.parseFloat(value);
            return Number.isFinite(parsed) ? parsed : null;
        };

        return {
            video_id: videoId,
            frame_n: parseNumber("meta-n", true),
            frame_idx: parseNumber("meta-idx", true),
            pts_time: parseNumber("meta-pts"),
            image_path: localImagePath(byId("detail-image").getAttribute("src")),
            query: (byId("query-input").value || "").trim(),
        };
    }

    function renderAttachment() {
        const preview = byId("chat-attachment-preview");
        preview.innerHTML = "";
        preview.classList.toggle("hidden", !attachedVideo);
        if (!attachedVideo) return;

        const label = document.createElement("span");
        label.textContent = `${attachedVideo.video_id} · ${attachedVideo.pts_time !== null ? `${attachedVideo.pts_time.toFixed(2)}s` : `frame ${attachedVideo.frame_n ?? "N/A"}`}`;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.setAttribute("aria-label", "Bỏ video đính kèm");
        remove.textContent = "×";
        remove.addEventListener("click", () => {
            attachedVideo = null;
            renderAttachment();
        });
        preview.append(label, remove);
    }

    function attachCurrentVideo() {
        attachedVideo = readCurrentVideo();
        if (!attachedVideo) {
            byId("chat-composer-hint").textContent = "Hãy chọn một keyframe trước khi share.";
            window.setTimeout(() => {
                byId("chat-composer-hint").textContent = "Enter để gửi · Shift+Enter xuống dòng";
            }, 2500);
            return;
        }
        renderAttachment();
        byId("chat-message-input").focus();
    }

    async function sendMessage() {
        const input = byId("chat-message-input");
        const text = input.value.trim();
        if (!text && !attachedVideo) return;

        const sendButton = byId("chat-send-button");
        sendButton.disabled = true;
        try {
            const response = await fetchWithTimeout(`${serverBaseUrl}/api/messages`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sender: displayName, text, video: attachedVideo }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
            appendMessage(data.message, false);
            input.value = "";
            attachedVideo = null;
            renderAttachment();
            setConnectionState(true, "Đã kết nối");
        } catch (error) {
            setConnectionState(false, "Gửi thất bại");
            byId("chat-composer-hint").textContent = error.message || "Không kết nối được host.";
        } finally {
            sendButton.disabled = false;
            input.focus();
        }
    }

    function saveSettings() {
        const nextServer = normalizeServerUrl(byId("chat-server-url").value);
        const nextName = byId("chat-display-name").value.trim().slice(0, 40) || defaultName();
        serverBaseUrl = nextServer;
        displayName = nextName;
        localStorage.setItem(STORAGE.server, serverBaseUrl);
        localStorage.setItem(STORAGE.name, displayName);
        byId("chat-server-url").value = serverBaseUrl;
        byId("chat-display-name").value = displayName;
        byId("chat-settings").classList.add("hidden");
        byId("chat-message-list").querySelectorAll(".chat-message-row").forEach((row) => row.remove());
        renderedMessageIds.clear();
        lastMessageId = 0;
        byId("chat-empty-state").classList.remove("hidden");
        setConnectionState(false, "Đang kết nối…");
        pollMessages(true);
    }

    function initialize() {
        serverBaseUrl = normalizeServerUrl(localStorage.getItem(STORAGE.server));
        displayName = localStorage.getItem(STORAGE.name) || defaultName();
        localStorage.setItem(STORAGE.name, displayName);
        byId("chat-server-url").value = serverBaseUrl;
        byId("chat-display-name").value = displayName;

        byId("team-chat-toggle").addEventListener("click", () => togglePanel());
        byId("chat-close-button").addEventListener("click", () => togglePanel(false));
        byId("chat-clear-button").addEventListener("click", clearOwnMessages);
        byId("chat-settings-button").addEventListener("click", () => {
            byId("chat-settings").classList.toggle("hidden");
        });
        byId("chat-save-settings").addEventListener("click", saveSettings);
        byId("chat-attach-button").addEventListener("click", attachCurrentVideo);
        byId("chat-send-button").addEventListener("click", sendMessage);
        byId("chat-message-input").addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && panelOpen) togglePanel(false);
        });

        window.AICShareChat = { open: () => togglePanel(true), attachCurrentVideo };
        pollMessages(true);
        window.setInterval(() => pollMessages(false), POLL_INTERVAL_MS);
    }

    document.addEventListener("DOMContentLoaded", initialize);
})();
