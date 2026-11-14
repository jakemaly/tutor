import { useState, type FormEvent } from "react";
import { useAui, useAuiState } from "@assistant-ui/react";
import { SendIcon } from "lucide-react";
import { Tldraw, useValue, type Editor } from "tldraw";
import "tldraw/tldraw.css";
import { buildWhiteboardPrompt, exportSelection } from "@/lib/whiteboard-snapshot";

const LICENSE_KEY = import.meta.env.VITE_TLDRAW_LICENSE_KEY;

type WhiteboardPanelProps = {
  /** Called after the selection was sent to the chat. */
  onSent?: () => void;
};

export default function WhiteboardPanel({ onSent }: WhiteboardPanelProps) {
  const [editor, setEditor] = useState<Editor | null>(null);

  return (
    <div className="whiteboard-panel" aria-label="Whiteboard">
      <div className="whiteboard-canvas">
        <Tldraw
          persistenceKey="tutor-whiteboard"
          licenseKey={LICENSE_KEY}
          onMount={setEditor}
        />
      </div>
      {editor && <AskBar editor={editor} onSent={onSent} />}
    </div>
  );
}

function AskBar({ editor, onSent }: { editor: Editor; onSent?: () => void }) {
  const aui = useAui();
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const selectedIds = useValue("selected shape ids", () => editor.getSelectedShapeIds(), [editor]);
  const [prompt, setPrompt] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasSelection = selectedIds.length > 0;
  const disabled = !hasSelection || isRunning || sending;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (disabled) return;
    setSending(true);
    setError(null);
    try {
      const { file, texts } = await exportSelection(editor, selectedIds);
      const composer = aui.composer();
      await composer.addAttachment(file);

      // The adapter reports upload failures as an errored attachment instead of throwing.
      const failed = composer
        .getState()
        .attachments.find(
          (attachment) =>
            attachment.name === file.name &&
            attachment.status.type === "incomplete" &&
            attachment.status.reason === "error",
        );
      if (failed) {
        await composer.attachment({ id: failed.id }).remove().catch(() => {});
        throw new Error(
          (failed.status.type === "incomplete" && failed.status.message) ||
            "Upload failed. Please try again.",
        );
      }

      composer.setText(buildWhiteboardPrompt(prompt, texts));
      composer.send();
      setPrompt("");
      onSent?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSending(false);
    }
  };

  return (
    <form className="whiteboard-ask" onSubmit={submit}>
      <input
        className="whiteboard-ask-input"
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder={
          hasSelection
            ? "Ask the tutor about the selection…"
            : "Select shapes on the board, then ask a question."
        }
        aria-label="Question about the selected whiteboard area"
      />
      <button
        type="submit"
        className="whiteboard-ask-send"
        disabled={disabled}
        aria-label="Send selection to tutor"
      >
        <SendIcon className="size-4" aria-hidden="true" />
        <span>{sending ? "Sending…" : "Ask tutor"}</span>
      </button>
      {error && (
        <p className="whiteboard-ask-error" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
