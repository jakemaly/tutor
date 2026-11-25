import { useMemo, useState } from "react";
import {
  AssistantRuntimeProvider,
  useAui,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import tmiLogo from "@/assets/TMI.png";
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
  useSidebar,
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
            <ChatControls onNewChat={startNewChat} />
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

function ChatControls({ onNewChat }: { onNewChat: () => void }) {
  const { open, openMobile, isMobile } = useSidebar();

  return (
    <header className="chat-controls">
      <SidebarTrigger
        aria-label="Toggle conversation list"
        aria-expanded={isMobile ? openMobile : open}
      />
      <div className="chat-controls-actions">
        <button className="new-chat-button" type="button" onClick={onNewChat}>
          <span aria-hidden="true">＋</span> New chat
        </button>
        <a className="tmi-home-link" href="/" aria-label="Tutor home">
          <img src={tmiLogo} alt="" width={36} height={36} />
        </a>
      </div>
    </header>
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
