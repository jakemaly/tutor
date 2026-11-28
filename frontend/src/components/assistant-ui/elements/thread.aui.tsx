import { groupAdjacentTools } from "./group-activity";
import { ActivityGroup, AssistantWaiting, SkillToolStatus, WebSearchToolStatus } from "./activity";
import { MarkdownText } from "@/components/assistant-ui/elements/markdown-text";
import { TooltipIconButton } from "@/components/assistant-ui/elements/tooltip-icon-button";
import {
  MessageDaySeparator,
  MessageTimestamp,
} from "@/components/day-separator";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/error-state";
import { cn } from "@/lib/utils";
import attachmentIcon from "@/assets/attachment.svg";
import microphoneIcon from "@/assets/microphone.svg";
import {
  ActionBarMorePrimitive,
  ActionBarPrimitive,
  AuiIf,
  AttachmentPrimitive,
  type AssistantState,
  BranchPickerPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiEvent,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  MoreHorizontalIcon,
  PencilIcon,
  FileTextIcon,
  ImageIcon,
  RefreshCwIcon,
  SquareIcon,
  XIcon,
} from "lucide-react";
import { useRef, useState, type FC, type Ref } from "react";

// Startup exposes a loading placeholder thread; treat it as a new chat so
// the composer mounts centered. Loads after startup keep the docked layout.
const isNewChatView = (s: AssistantState) =>
  s.thread.messages.length === 0 &&
  (!s.thread.isLoading || s.threads.isLoading);

// A switched thread that is still fetching its history: skeleton, not welcome.
const isHistoryLoadingView = (s: AssistantState) =>
  s.thread.messages.length === 0 &&
  s.thread.isLoading &&
  !s.thread.isDisabled &&
  !s.threads.isLoading;

const ThreadHistorySkeleton: FC = () => (
  <div
    data-slot="aui_thread-history-skeleton"
    role="status"
    className="animate-in fade-in fill-mode-both flex flex-col [animation-delay:150ms] animation-duration-[200ms]"
  >
    <span className="sr-only">Loading conversation</span>
    <div className="flex animate-pulse flex-col gap-y-6 motion-reduce:animate-none">
      <div className="bg-muted ml-auto h-9 w-2/5 rounded-xl" />
      <div className="flex flex-col gap-y-2">
        <div className="bg-muted h-4 w-11/12 rounded-md" />
        <div className="bg-muted h-4 w-4/5 rounded-md" />
        <div className="bg-muted h-4 w-3/5 rounded-md" />
      </div>
      <div className="bg-muted ml-auto h-9 w-1/3 rounded-xl" />
      <div className="flex flex-col gap-y-2">
        <div className="bg-muted h-4 w-10/12 rounded-md" />
        <div className="bg-muted h-4 w-2/3 rounded-md" />
      </div>
    </div>
  </div>
);

export const Thread: FC = () => {
  const isEmpty = useAuiState(isNewChatView);
  const composerInput = useRef<HTMLTextAreaElement>(null);

  return (
    <ThreadPrimitive.Root
      className="aui-root aui-thread-root bg-background @container flex h-full min-h-0 flex-col"
    >
      <ThreadPrimitive.Viewport
        turnAnchor="top"
        data-slot="aui_thread-viewport"
        className="aui-thread-viewport relative flex min-h-0 flex-1 flex-col overflow-auto"
      >
        <div
          className={cn(
            "aui-thread-content",
            isEmpty && "aui-thread-content-home",
          )}
        >
          {isEmpty && <h1 className="aui-thread-welcome">What can I help you with?</h1>}
          <AuiIf condition={isHistoryLoadingView}>
            <ThreadHistorySkeleton />
          </AuiIf>

          <div
            data-slot="aui_message-group"
            className="aui-message-group"
          >
            <ThreadPrimitive.Messages>
              {() => <ThreadMessage />}
            </ThreadPrimitive.Messages>
          </div>

          <ThreadPrimitive.ViewportFooter
            className={cn(
              "aui-thread-viewport-footer",
              isEmpty && "aui-empty-composer",
              !isEmpty && "aui-thread-viewport-footer-docked",
            )}
          >
            {!isEmpty && <ThreadScrollToBottom />}
            <Composer isEmpty={isEmpty} inputRef={composerInput} />
            {!isEmpty && (
              <p className="aui-disclaimer">
                Tutor can make mistakes. Check important work with your course materials.
              </p>
            )}
          </ThreadPrimitive.ViewportFooter>
          {isEmpty && <PromptSuggestions onSelect={() => composerInput.current?.focus()} />}
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadMessage: FC = () => {
  const role = useAuiState((s) => s.message.role);
  const isEditing = useAuiState((s) => s.message.composer.isEditing);

  if (isEditing) return <EditComposer />;
  if (role === "user") return <UserMessage />;
  return <AssistantMessage />;
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="aui-thread-scroll-to-bottom dark:border-border dark:bg-background dark:hover:bg-accent absolute -top-12 z-10 self-center rounded-full p-4 disabled:invisible"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const promptRows = [
  ["Check my reasoning on a problem", "Help me make a study plan"],
  ["Explain a concept step by step", "Quiz me on a topic"],
];

const PromptSuggestions: FC<{ onSelect: () => void }> = ({ onSelect }) => {
  return (
    <div className="aui-prompt-suggestions" role="group" aria-label="Suggested starting points">
      {promptRows.map((row, rowIndex) => (
        <div key={row[0]} className="aui-prompt-row">
          {row.map((prompt, columnIndex) => (
            <ThreadPrimitive.Suggestion
              key={prompt}
              prompt={prompt}
              send={false}
              onClick={onSelect}
              style={{ animationDelay: `${120 + (rowIndex * 2 + columnIndex) * 70}ms` }}
            >
              {prompt}
            </ThreadPrimitive.Suggestion>
          ))}
        </div>
      ))}
    </div>
  );
};

const Composer: FC<{ isEmpty: boolean; inputRef: Ref<HTMLTextAreaElement> }> = ({
  isEmpty,
  inputRef,
}) => {
  return (
    <ComposerPrimitive.Root className="aui-composer-root">
      <div data-slot="aui_composer-shell" className="aui-composer-shell">
        <ComposerPrimitive.Attachments>
          {({ attachment }) => (
            <AttachmentPrimitive.Root
              className="flex min-w-0 items-center gap-2 rounded-lg border bg-background px-2.5 py-2"
              key={attachment.id}
            >
              {attachment.type === "image" ? (
                <ImageIcon className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
              ) : (
                <FileTextIcon className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
              )}
              <span className="min-w-0 flex-1 truncate text-xs font-medium">
                <AttachmentPrimitive.Name />
              </span>
              <AttachmentPrimitive.Remove aria-label={`Remove ${attachment.name}`}>
                <XIcon className="size-4" />
              </AttachmentPrimitive.Remove>
            </AttachmentPrimitive.Root>
          )}
        </ComposerPrimitive.Attachments>
        <AttachmentAddError />
        <ComposerPrimitive.Input
          ref={inputRef}
          placeholder={isEmpty ? "Share a question or an attempt…" : "Reply to your tutor…"}
          className="aui-composer-input"
          rows={1}
          aria-label="Message Tutor"
        />
        <ComposerAction />
      </div>
    </ComposerPrimitive.Root>
  );
};

const AttachmentAddError: FC = () => {
  const [error, setError] = useState<string | null>(null);
  useAuiEvent("composer.attachmentAddError", ({ message }) => setError(message));
  useAuiEvent("composer.attachmentAdd", () => setError(null));
  return error ? <p className="px-2 text-xs text-destructive" role="alert">{error}</p> : null;
};

const ComposerAction: FC = () => {
  // Submission may be pending before the model run starts.
  const isSending = useAuiState(
    (s) =>
      s.composer.submission !== undefined &&
      !(s.thread.isRunning && s.thread.capabilities.cancel),
  );

  return (
    <div className="aui-composer-actions">
      <ComposerPrimitive.AddAttachment asChild>
        <TooltipIconButton
          tooltip="Attach a PDF or image"
          aria-label="Attach a PDF or image"
          className="aui-composer-attachment"
          side="top"
        >
          <img src={attachmentIcon} alt="" width={24} height={24} />
        </TooltipIconButton>
      </ComposerPrimitive.AddAttachment>
      <div className="aui-composer-submit">
        <button
          className="aui-composer-voice"
          type="button"
          disabled
          aria-label="Voice input is unavailable"
          title="Voice input is unavailable"
        >
          <img src={microphoneIcon} alt="" width={22} height={22} />
        </button>
        <AuiIf
          condition={(s) =>
            !s.composer.canCancel ||
            (s.thread.voice !== undefined &&
              s.composer.submission === undefined)
          }
        >
          <ComposerPrimitive.Send asChild>
            <TooltipIconButton
              tooltip="Send message"
              side="top"
              type="button"
              variant="default"
              size="icon"
              className="aui-composer-send"
              aria-label="Send message"
            >
              <ArrowUpIcon aria-hidden="true" className="size-[18px]" strokeWidth={1.8} />
            </TooltipIconButton>
          </ComposerPrimitive.Send>
        </AuiIf>
        <AuiIf
          condition={(s) =>
            s.composer.canCancel &&
            (s.thread.voice === undefined ||
              s.composer.submission !== undefined)
          }
        >
          <ComposerPrimitive.Cancel asChild>
            <Button
              type="button"
              variant="default"
              size="icon"
              className="aui-composer-cancel"
              aria-label={isSending ? "Cancel sending" : "Stop generating"}
            >
              <SquareIcon className="aui-composer-cancel-icon size-3.5 fill-current" />
            </Button>
          </ComposerPrimitive.Cancel>
        </AuiIf>
      </div>
    </div>
  );
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      data-slot="aui_assistant-message-root"
      data-role="assistant"
      className="group/message-root fade-in slide-in-from-bottom-1 animate-in relative -mb-7.5 pb-7.5 duration-150"
    >
      <MessageDaySeparator />
      <div
        data-slot="aui_assistant-message-content"
        className="wrap-break-word"
      >
        <MessageTimestamp />
        <MessagePrimitive.Unstable_PartsGrouped
          groupingFunction={groupAdjacentTools}
          components={{
            Text: MarkdownText,
            Group: ActivityGroup,
            tools: { by_name: { load_skill: SkillToolStatus, web_search: WebSearchToolStatus } },
          }}
        />
        <AssistantWaiting />
      </div>

      <AssistantError />

      <div
        data-slot="aui_assistant-message-footer"
        className="aui-assistant-footer flex items-center"
      >
        <BranchPicker />
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantError: FC = () => {
  const aui = useAui();
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const error = useAuiState((s) =>
    s.message.status?.type === "incomplete" &&
      s.message.status.reason === "error"
      ? s.message.status.error
      : undefined,
  );

  const detail =
    typeof error === "string" && error.trim().length > 0
      ? error
      : "The response could not be completed. Please try again.";

  return (
    <MessagePrimitive.Error>
      <div className="mt-2">
        <ErrorState
          title="Something went wrong"
          detail={detail}
          retrying={isRunning}
          onRetry={() => aui.message().reload()}
        />
      </div>
    </MessagePrimitive.Error>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="never"
      className="aui-assistant-action-bar-root flex items-center gap-1 text-muted-foreground"
    >
      <ActionBarPrimitive.Copy asChild>
        <TooltipIconButton tooltip="Copy response" className="aui-message-action">
          <AuiIf condition={(s) => s.message.isCopied}>
            <CheckIcon className="animate-in zoom-in-50 fade-in duration-200 ease-out" />
          </AuiIf>
          <AuiIf condition={(s) => !s.message.isCopied}>
            <CopyIcon className="animate-in zoom-in-75 fade-in duration-150" />
          </AuiIf>
          <span aria-hidden="true">Copy</span>
        </TooltipIconButton>
      </ActionBarPrimitive.Copy>
      <ActionBarPrimitive.Reload asChild>
        <TooltipIconButton tooltip="Regenerate response" className="aui-message-action">
          <RefreshCwIcon />
          <span aria-hidden="true">Regenerate</span>
        </TooltipIconButton>
      </ActionBarPrimitive.Reload>
      <ActionBarMorePrimitive.Root>
        <ActionBarMorePrimitive.Trigger asChild>
          <TooltipIconButton
            tooltip="More message actions"
            className="data-[state=open]:bg-accent"
          >
            <MoreHorizontalIcon />
          </TooltipIconButton>
        </ActionBarMorePrimitive.Trigger>
        <ActionBarMorePrimitive.Content
          side="bottom"
          align="start"
          sideOffset={6}
          className="aui-action-bar-more-content bg-popover/95 text-popover-foreground data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=open]:animate-in data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=closed]:animate-out data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 z-50 min-w-32 overflow-hidden rounded-xl border p-1.5 shadow-lg backdrop-blur-sm"
        >
          <ActionBarPrimitive.ExportMarkdown asChild>
            <ActionBarMorePrimitive.Item className="aui-action-bar-more-item hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-1.5 text-sm outline-none select-none">
              <DownloadIcon className="size-4" />
              Export as Markdown
            </ActionBarMorePrimitive.Item>
          </ActionBarPrimitive.ExportMarkdown>
        </ActionBarMorePrimitive.Content>
      </ActionBarMorePrimitive.Root>
    </ActionBarPrimitive.Root>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      data-slot="aui_user-message-root"
      className="group/message-root fade-in slide-in-from-bottom-1 animate-in grid grid-cols-[minmax(36px,1fr)_auto] content-start duration-150 [&:where(>*)]:col-start-2"
      data-role="user"
    >
      <MessageDaySeparator />
      <div className="aui-user-message-content-wrapper relative col-start-2 min-w-0">
        <div className="aui-user-message-content peer wrap-break-word empty:hidden">
          <MessagePrimitive.Parts />
        </div>
        <MessageTimestamp align="end" />
        <MessagePrimitive.Attachments>
          {({ attachment }) => (
            <a
              href={`/api/attachments/${encodeURIComponent(attachment.id)}`}
              target="_blank"
              rel="noreferrer"
              className="mt-2 flex max-w-64 items-center gap-2 rounded-lg border bg-background p-2 text-xs hover:bg-muted"
            >
              {attachment.type === "image" ? <ImageIcon className="size-4" /> : <FileTextIcon className="size-4" />}
              <span className="min-w-0 truncate">{attachment.name}</span>
            </a>
          )}
        </MessagePrimitive.Attachments>
        <div className="aui-user-action-bar-wrapper pointer-events-none absolute inset-s-0 top-1/2 -translate-x-full -translate-y-1/2 pe-2 opacity-0 transition-opacity group-focus-within/message-root:pointer-events-auto group-focus-within/message-root:opacity-100 group-hover/message-root:pointer-events-auto group-hover/message-root:opacity-100 peer-empty:hidden rtl:translate-x-full">
          <UserActionBar />
        </div>
      </div>

      <BranchPicker
        data-slot="aui_user-branch-picker"
        className="col-span-full col-start-1 -me-1 justify-end"
      />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="never"
      className="aui-user-action-bar-root flex flex-col items-end"
    >
      <ActionBarPrimitive.Edit asChild>
        <TooltipIconButton tooltip="Edit message" className="aui-user-action-edit">
          <PencilIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Edit>
    </ActionBarPrimitive.Root>
  );
};

const EditComposer: FC = () => {
  return (
    <MessagePrimitive.Root
      data-slot="aui_edit-composer-wrapper"
      className="flex flex-col px-2 [contain-intrinsic-size:auto_200px] [content-visibility:auto]"
    >
      <ComposerPrimitive.Root className="aui-edit-composer-root border-foreground/10 focus-within:border-foreground/25 ms-auto flex w-full max-w-[85%] cursor-text flex-col rounded-(--composer-radius) border bg-(--composer-bg) shadow-[0_4px_16px_-8px_rgba(0,0,0,0.08),0_1px_2px_rgba(0,0,0,0.04)] transition-[border-color] dark:shadow-none">
        <ComposerPrimitive.Input
          aria-label="Edit message"
          name="edited-message"
          className="aui-edit-composer-input text-foreground min-h-14 w-full resize-none bg-transparent px-4 pt-3 pb-1 text-base outline-none"
          autoFocus
        />
        <div className="aui-edit-composer-footer mx-2.5 mb-2.5 flex items-center gap-1.5 self-end">
          <ComposerPrimitive.Cancel asChild>
            <Button variant="ghost" size="sm" className="h-8 px-3">
              Cancel
            </Button>
          </ComposerPrimitive.Cancel>
          <ComposerPrimitive.Send asChild>
            <Button size="sm" className="h-8 px-3">
              Save
            </Button>
          </ComposerPrimitive.Send>
        </div>
      </ComposerPrimitive.Root>
    </MessagePrimitive.Root>
  );
};

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({
  className,
  ...rest
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={cn(
        "aui-branch-picker-root text-muted-foreground -ms-2 me-2 inline-flex items-center text-xs",
        className,
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous asChild>
        <TooltipIconButton tooltip="Previous response">
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="aui-branch-picker-state font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton tooltip="Next response">
          <ChevronRightIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};
