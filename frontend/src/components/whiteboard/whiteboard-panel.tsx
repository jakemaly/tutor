import { useEffect, useId, useState, type FormEvent } from "react";
import { useAui, useAuiState } from "@assistant-ui/react";
import { SendIcon } from "lucide-react";
import {
  ArrowDownToolbarItem,
  ArrowLeftToolbarItem,
  ArrowRightToolbarItem,
  ArrowToolbarItem,
  ArrowUpToolbarItem,
  AssetToolbarItem,
  CheckBoxToolbarItem,
  CloudToolbarItem,
  DefaultToolbar,
  DiamondToolbarItem,
  DrawToolbarItem,
  EllipseToolbarItem,
  EraserToolbarItem,
  FrameToolbarItem,
  HandToolbarItem,
  HeartToolbarItem,
  HexagonToolbarItem,
  HighlightToolbarItem,
  LaserToolbarItem,
  LineToolbarItem,
  NoteToolbarItem,
  OvalToolbarItem,
  PORTRAIT_BREAKPOINT,
  RectangleToolbarItem,
  RhombusToolbarItem,
  SelectToolbarItem,
  StarToolbarItem,
  TextToolbarItem,
  Tldraw,
  TldrawUiButton,
  TldrawUiButtonIcon,
  TriangleToolbarItem,
  XBoxToolbarItem,
  useBreakpoint,
  useEditor,
  useTools,
  useValue,
  type Editor,
  type TLComponents,
  type TLEventInfo,
} from "tldraw";
import "tldraw/tldraw.css";
import { buildWhiteboardPrompt, exportSelection } from "@/lib/whiteboard-snapshot";

const LICENSE_KEY = import.meta.env.VITE_TLDRAW_LICENSE_KEY;

/** Distance between neighbouring dots in page units at 100% zoom. */
const DOT_SPACING = 24;

/** Isometric (triangular) dot lattice that pans and zooms with the canvas. */
function IsometricDotBackground() {
  const editor = useEditor();
  const patternId = `whiteboard-dots-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const camera = useValue("camera", () => editor.getCamera(), [editor]);

  // Double or halve the spacing so the dots stay readable at any zoom level.
  let spacing = DOT_SPACING * camera.z;
  while (spacing < 12) spacing *= 2;
  while (spacing > 48) spacing /= 2;
  const rowPair = spacing * Math.sqrt(3);

  return (
    <div className="tl-background">
      <svg className="whiteboard-dots" aria-hidden="true">
        <defs>
          <pattern
            id={patternId}
            width={spacing}
            height={rowPair}
            patternUnits="userSpaceOnUse"
            patternTransform={`translate(${camera.x * camera.z} ${camera.y * camera.z})`}
          >
            <circle cx={0} cy={0} r={1.25} />
            <circle cx={spacing} cy={0} r={1.25} />
            <circle cx={spacing / 2} cy={rowPair / 2} r={1.25} />
            <circle cx={0} cy={rowPair} r={1.25} />
            <circle cx={spacing} cy={rowPair} r={1.25} />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill={`url(#${patternId})`} />
      </svg>
    </div>
  );
}

/** Tools shown directly in the phone tool list; the shapes after them fold into "more". */
const MOBILE_MAIN_TOOLS = 8;

/** Tldraw's default tools, reordered so the most used ones sit at the top on phones. */
function MobileToolbarContent() {
  return (
    <>
      <SelectToolbarItem />
      <TextToolbarItem />
      <DrawToolbarItem />
      <EraserToolbarItem />
      <HandToolbarItem />
      <ArrowToolbarItem />
      <NoteToolbarItem />
      <AssetToolbarItem />

      <RectangleToolbarItem />
      <EllipseToolbarItem />
      <TriangleToolbarItem />
      <DiamondToolbarItem />

      <HexagonToolbarItem />
      <OvalToolbarItem />
      <RhombusToolbarItem />
      <StarToolbarItem />

      <CloudToolbarItem />
      <HeartToolbarItem />
      <XBoxToolbarItem />
      <CheckBoxToolbarItem />

      <ArrowLeftToolbarItem />
      <ArrowUpToolbarItem />
      <ArrowDownToolbarItem />
      <ArrowRightToolbarItem />

      <LineToolbarItem />
      <HighlightToolbarItem />
      <LaserToolbarItem />
      <FrameToolbarItem />
    </>
  );
}

/** Phone-sized boards collapse the tool bar into a single top-left button. */
function WhiteboardToolbar() {
  const breakpoint = useBreakpoint();
  if (breakpoint < PORTRAIT_BREAKPOINT.MOBILE_SM) return <MobileToolbar />;
  // Show the tool bar as a vertical column on the left of the board instead of along the bottom.
  return <DefaultToolbar orientation="vertical" />;
}

function MobileToolbar() {
  const editor = useEditor();
  const tools = useTools();
  const [open, setOpen] = useState(false);
  const activeToolId = useValue("current tool id", () => editor.getCurrentToolId(), [editor]);
  const activeIcon = tools[activeToolId]?.icon ?? "tool-pointer";

  // Retract the tool list as soon as the learner touches the board itself.
  useEffect(() => {
    if (!open) return;
    const handleEvent = (info: TLEventInfo) => {
      if (info.type === "pointer" && info.name === "pointer_down") setOpen(false);
    };
    editor.on("event", handleEvent);
    return () => {
      editor.off("event", handleEvent);
    };
  }, [editor, open]);

  return (
    <div className="whiteboard-mobile-toolbar">
      <TldrawUiButton
        type="icon"
        className="whiteboard-mobile-toolbar-trigger"
        aria-label="Tools"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <TldrawUiButtonIcon icon={activeIcon} />
      </TldrawUiButton>
      {open && (
        <DefaultToolbar
          orientation="vertical"
          minItems={MOBILE_MAIN_TOOLS}
          maxItems={MOBILE_MAIN_TOOLS}
        >
          <MobileToolbarContent />
        </DefaultToolbar>
      )}
    </div>
  );
}

const COMPONENTS: TLComponents = {
  Toolbar: WhiteboardToolbar,
  Background: IsometricDotBackground,
};

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
          components={COMPONENTS}
          onMount={(mounted) => {
            // Fit the saved drawing in view (never zooming past 100%) instead of restoring the
            // last camera offset. Wait a frame so the viewport size has been measured.
            requestAnimationFrame(() => {
              const bounds = mounted.getCurrentPageBounds();
              if (bounds) mounted.zoomToBounds(bounds, { targetZoom: 1, inset: 32 });
            });
            setEditor(mounted);
          }}
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
