import { useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";

export default function App() {
  // Remounting gives New chat a fresh runtime; saved threads are a later workstream.
  const [session, setSession] = useState(0);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Tutor home">Tutor</a>
        <button className="new-chat-button" onClick={() => setSession((value) => value + 1)}>
          <span aria-hidden="true">＋</span> New chat
        </button>
      </header>
      <ChatSession key={session} />
    </main>
  );
}

function ChatSession() {
  const [error, setError] = useState<string | null>(null);
  const runtime = useDataStreamRuntime({
    api: "/api/chat",
    protocol: "data-stream",
    onResponse: (response) => {
      if (response.ok) setError(null);
    },
    onError: (cause) => setError(cause.message),
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <section className="chat-panel" aria-label="Chat">
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button className="error-dismiss" onClick={() => setError(null)} aria-label="Dismiss error">×</button>
          </div>
        )}
        <Thread />
      </section>
    </AssistantRuntimeProvider>
  );
}
