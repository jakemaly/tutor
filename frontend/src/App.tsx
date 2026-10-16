// React state resets the chat when New chat is selected without persisting history.
import { useState } from "react";
// assistant-ui primitives supply chat state and accessible message/composer components.
import {
  AuiIf,
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
// This runtime connects assistant-ui to the FastAPI data-stream endpoint.
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
// Render streamed assistant text as Markdown with GitHub-style tables and lists.
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";

export default function App() {
  // The key below remounts ChatSession, which gives each new chat empty in-memory history.
  const [session, setSession] = useState(0);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Tutor home">
          Tutor
        </a>
        <button className="new-chat-button" onClick={() => setSession((value) => value + 1)}>
          <span aria-hidden="true">＋</span> New chat
        </button>
      </header>
      <ChatSession key={session} />
    </main>
  );
}

// Own the per-chat runtime and keep request errors available to the visible chat screen.
function ChatSession() {
  const [error, setError] = useState<string | null>(null);
  const runtime = useDataStreamRuntime({
    api: "/api/chat",
    protocol: "data-stream",
    // Successful responses clear old errors; failed requests are shown above the conversation.
    onResponse: (response) => {
      if (response.ok) setError(null);
    },
    onError: (cause) => setError(cause.message),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatContent error={error} clearError={() => setError(null)} />
    </AssistantRuntimeProvider>
  );
}

// Arrange empty/populated conversation states and connect Stop to assistant-ui's runtime.
function ChatContent({ error, clearError }: { error: string | null; clearError: () => void }) {
  const aui = useAui();
  const isRunning = useAuiState((state) => state.thread.isRunning);

  return (
    <section className="chat-panel" aria-label="Chat">
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button className="error-dismiss" onClick={clearError} aria-label="Dismiss error">×</button>
        </div>
      )}

      <ThreadPrimitive.Root className="thread-root">
        {/* The empty view centers the first prompt; populated chats use a scrolling viewport. */}
        <AuiIf condition={(state) => state.thread.isEmpty}>
          <div className="empty-state">
            <div className="empty-state-content">
              <h1>Where should we begin?</h1>
              <Composer isRunning={isRunning} stop={() => aui.thread().cancelRun()} />
              <p className="composer-disclaimer">Tutor can make mistakes. Check important information.</p>
            </div>
          </div>
        </AuiIf>

        <AuiIf condition={(state) => !state.thread.isEmpty}>
          <ThreadPrimitive.Viewport className="thread-viewport" autoScroll>
            <div className="message-list">
              <ThreadPrimitive.Messages>
                {/* Select a visual component by role while assistant-ui owns the message data. */}
                {({ message }) => message.role === "user" ? <UserMessage /> : <AssistantMessage />}
              </ThreadPrimitive.Messages>
            </div>

            <ThreadPrimitive.ViewportFooter className="thread-footer">
              <Composer isRunning={isRunning} stop={() => aui.thread().cancelRun()} />
              <p className="composer-disclaimer">Tutor can make mistakes. Check important information.</p>
            </ThreadPrimitive.ViewportFooter>
          </ThreadPrimitive.Viewport>
        </AuiIf>
      </ThreadPrimitive.Root>
    </section>
  );
}

function Composer({ isRunning, stop }: { isRunning: boolean; stop: () => void }) {
  // Replace Send with Stop during a stream so the user can cancel the active run.
  return (
    <ComposerPrimitive.Root className="composer">
      <ComposerPrimitive.Input
        aria-label="Message"
        placeholder="Message Tutor"
        className="composer-input"
        rows={1}
        submitMode="enter"
      />
      <div className="composer-actions">
        {isRunning ? (
          <button
            className="send-button stop-button"
            type="button"
            onClick={stop}
            aria-label="Stop response"
          >
            <span className="stop-icon" aria-hidden="true" />
          </button>
        ) : (
          <ComposerPrimitive.Send className="send-button" aria-label="Send message">
            <span aria-hidden="true">↑</span>
          </ComposerPrimitive.Send>
        )}
      </div>
    </ComposerPrimitive.Root>
  );
}

function UserMessage() {
  // User text stays plain and is aligned to the right by the corresponding CSS classes.
  return (
    <MessagePrimitive.Root className="message-row user-row">
      <div className="user-bubble"><MessagePrimitive.Parts /></div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  // Only assistant output is Markdown-rendered; user-entered text remains literal.
  return (
    <MessagePrimitive.Root className="message-row assistant-row">
      <div className="assistant-content">
        <MessagePrimitive.Parts
          components={{ Text: () => <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} /> }}
        />
      </div>
    </MessagePrimitive.Root>
  );
}
