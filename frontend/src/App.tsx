import { useMemo, useState } from "react";
import {
  AssistantRuntimeProvider,
  useAui,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
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

export default function App() {
  const [error, setError] = useState<string | null>(null);
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
              <button className="new-chat-button" onClick={startNewChat}>
                <span aria-hidden="true">＋</span> New chat
              </button>
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
            <section className="chat-panel" aria-label="Chat">
              <Thread />
            </section>
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
