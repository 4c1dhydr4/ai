
const chatMessages = document.getElementById("chatMessages");
const messageInput = document.getElementById("messageInput");
const chatForm = document.getElementById("chatForm");
const sendButton = document.getElementById("sendButton");

// Configure Markdown.
marked.setOptions({
    gfm: true,
    breaks: true,
});

// Scroll to bottom.
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Render Markdown safely.
function renderMarkdown(element, content) {

    if (!requiresMarkdown(content)) {
        element.textContent = content;
        return;
    }

    const renderedHTML = marked.parse(content);

    element.innerHTML = DOMPurify.sanitize(renderedHTML);
}

function requiresMarkdown(content) {

    const patterns = [
        /^#{1,6}\s+/m,                  // Headers
        /^\s*[-*+]\s+/m,                // Unordered lists
        /^\s*\d+\.\s+/m,                // Ordered lists
        /^\s*>\s+/m,                    // Blockquotes
        /```[\s\S]*?```/,               // Code blocks
        /`[^`]+`/,                      // Inline code
        /\*\*[^*]+\*\*/,                // Bold
        /__[^_]+__/,                    // Bold
        /(?<!\*)\*[^*\n]+\*(?!\*)/,     // Italic
        /^\s*\|.+\|\s*$/m,              // Tables
        /\[.+?\]\(.+?\)/,               // Links
        /!\[.*?\]\(.+?\)/,              // Images
        /^\s*([-*_])\1{2,}\s*$/m,       // Horizontal rules
    ];

    return patterns.some(pattern => pattern.test(content));
}

// Render all messages already loaded by Django.
function renderExistingMessages() {

    const messages = chatMessages.querySelectorAll(
        ".message-content"
    );

    messages.forEach((message) => {

        const content = message.textContent;

        // Render Markdown only for assistant messages.
        if (message.closest(".message.assistant")) {
            renderMarkdown(message, content);
        }

    });

}

// Create a new message.
function createMessage(role, content = "") {

    const message = document.createElement("div");

    message.className = `message ${role}`;

    const bubble = document.createElement("div");

    bubble.className = "message-content";

    if (role === "assistant" && content) {
        renderMarkdown(bubble, content);
    } else {
        bubble.textContent = content;
    }

    message.appendChild(bubble);

    chatMessages.appendChild(message);

    return bubble;
}

// Initialize existing conversation.
renderExistingMessages();
scrollToBottom();

// Automatically expand textarea.
messageInput.addEventListener("input", function () {

    this.style.height = "auto";

    this.style.height = `${this.scrollHeight}px`;

});

// Submit message and consume streaming response.
chatForm.addEventListener("submit", async function (event) {

    event.preventDefault();

    const message = messageInput.value.trim();

    if (!message) return;

    // Clear empty state.
    const emptyState = chatMessages.querySelector(".empty-state");

    if (emptyState) {
        emptyState.remove();
    }

    // Display user's message immediately.
    createMessage("user", message);

    // Prepare assistant response bubble.
    const assistantBubble = createMessage("assistant");

    assistantBubble.classList.add("assistant-message");

    const formData = new FormData(chatForm);

    // Disable form while generating.
    sendButton.disabled = true;
    sendButton.textContent = "Generating...";

    messageInput.disabled = true;
    messageInput.value = "";
    messageInput.style.height = "auto";

    try {

        const response = await fetch(chatForm.action, {
            method: "POST",
            body: formData,
            headers: {
                "X-Requested-With": "XMLHttpRequest",
            },
        });

        if (!response.ok) {

            const error = await response.text();

            throw new Error(
                `HTTP ${response.status}: ${error.substring(0, 300)}`
            );

        }

        if (response.redirected) {

            throw new Error(
                `Unexpected redirect to: ${response.url}`
            );

        }

        const contentType = response.headers.get("Content-Type") || "";

        if (!contentType.includes("text/plain")) {

            const body = await response.text();

            throw new Error(
                `Expected text/plain but received ${contentType}. ` +
                `Response: ${body.substring(0, 300)}`
            );

        }

        if (!response.body) {
            throw new Error("Streaming is not supported.");
        }

        // Update conversation URL and form destination.
        const conversationUrl = response.headers.get(
            "X-Conversation-URL"
        );

        if (conversationUrl) {

            window.history.replaceState(
                {},
                "",
                conversationUrl
            );

            chatForm.action = conversationUrl;

        }

        const reader = response.body.getReader();

        const decoder = new TextDecoder("utf-8");

        let assistantResponse = "";

        while (true) {

            const { value, done } = await reader.read();

            if (done) break;

            assistantResponse += decoder.decode(value, {
                stream: true,
            });

            // Temporary plain-text display while generating.
            assistantBubble.textContent = assistantResponse;

            scrollToBottom();

        }

        // Flush remaining decoder data.
        assistantResponse += decoder.decode();

        // Render completed assistant response as Markdown.
        renderMarkdown(
            assistantBubble,
            assistantResponse
        );

        scrollToBottom();

    } catch (error) {

        console.error("Chat streaming error:", error);

        assistantBubble.textContent += (
            "\n\nUnable to complete the request. " +
            "Please try again."
        );

    } finally {

        messageInput.disabled = false;
        sendButton.disabled = false;
        sendButton.textContent = "Send";

        messageInput.focus();

        scrollToBottom();

    }

});
