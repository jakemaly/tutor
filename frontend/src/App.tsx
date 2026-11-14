import { lazy, Suspense, useMemo, useState } from "react";
import { ArrowLeftIcon, PenLineIcon } from "lucide-react";
import {
  AssistantRuntimeProvider,
  useAui,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import { attachmentAdapter } from "@/lib/attachments";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { ThreadListSidebar } from "@/components/threadlist-sidebar.aui";
import {
  SessionThreadHistoryAdapter,
  SessionThreadListAdapter,
} from "@/lib/session-thread-history";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";

// tldraw is large, so it loads only when the learner opens the whiteboard.
const WhiteboardPanel = lazy(() => import("@/components/whiteboard/whiteboard-panel"));

export default function App() {
  const [error, setError] = useState<string | null>(null);
  const [whiteboardOpen, setWhiteboardOpen] = useState(false);
  const adapter = useMemo(
    () =>
      Object.assign(new SessionThreadListAdapter(), {
        unstable_useAdapters: useSessionRuntimeAdapters,
      }),
    [],
  );
  const runtime = useRemoteThreadListRuntime({
    adapter,
    runtimeHook: function useTutorRuntime() {
      return useDataStreamRuntime({
        api: "/api/chat",
        protocol: "data-stream",
        adapters: { attachments: attachmentAdapter },
        onResponse: (response) => {
          if (response.ok) setError(null);
        },
        onError: (cause) => setError(cause.message),
      });
    },
  });

  const startNewChat = () => {
    setError(null);
    void runtime.threads.switchToNewThread().catch((cause: unknown) => {
      setError(cause instanceof Error ? cause.message : String(cause));
    });
  };

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <SidebarProvider className="app-shell">
          <ThreadListSidebar />
          <SidebarInset className="app-main">
            <header className="topbar">
              <div className="topbar-leading">
                <SidebarTrigger aria-label="Toggle conversation list" />
                <a className="brand" href="/" aria-label="Tutor home">
                  Tutor
                </a>
              </div>
              <div className="topbar-actions">
                <button
                  className="new-chat-button"
                  onClick={() => setWhiteboardOpen((open) => !open)}
                  aria-pressed={whiteboardOpen}
                >
                  {whiteboardOpen ? (
                    <ArrowLeftIcon className="inline size-3.5" aria-hidden="true" />
                  ) : (
                    <PenLineIcon className="inline size-3.5" aria-hidden="true" />
                  )}{" "}
                  {whiteboardOpen ? "Back to Tutor" : "Whiteboard"}
                </button>
                <button className="new-chat-button" onClick={startNewChat}>
                  <span aria-hidden="true">＋</span> New chat
                </button>
              </div>
            </header>
            {error && (
              <div className="error-banner" role="alert">
                <span>{error}</span>
                <button
                  className="error-dismiss"
                  onClick={() => setError(null)}
                  aria-label="Dismiss error"
                >
                  ×
                </button>
              </div>
            )}
            <div className="workspace-split">
              {whiteboardOpen && (
                <Suspense
                  fallback={<div className="whiteboard-panel whiteboard-loading">Loading whiteboard…</div>}
                >
                  <WhiteboardPanel onSent={() => setWhiteboardOpen(false)} />
                </Suspense>
              )}
              <section
                className="chat-panel"
                aria-label="Chat"
                hidden={whiteboardOpen}
              >
                <Thread />
              </section>
            </div>
          </SidebarInset>
        </SidebarProvider>
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}

function useSessionRuntimeAdapters() {
  const aui = useAui();
  const history = useMemo(
    () => {
      const threadListItem = aui.threadListItem;
      return new SessionThreadHistoryAdapter(() => {
        const state = threadListItem.getState();
        return state.remoteId ?? state.id;
      });
    },
    [aui],
  );

  return { history };
}
