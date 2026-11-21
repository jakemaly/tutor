import {
  defineToolkit,
  useAui,
  useAuiState,
  type ToolCallMessagePartComponent,
} from "@assistant-ui/react";
import { SafeContentFrame } from "safe-content-frame";
import { useEffect, useId, useRef, useState } from "react";

type StudyOption = {
  id: string;
  label: string;
};

type StudyElementArgs = {
  kind: "flashcard" | "mcq" | "true_false" | "multi_select";
  question: string;
  options: StudyOption[];
};

type PresentedResult = {
  presented: boolean;
};

type HtmlPreviewArgs = {
  title: string;
  html: string;
};

const MAX_PREVIEW_HTML_LENGTH = 28_000;
const htmlFrame = new SafeContentFrame("socratic-ai-tutor");

const StudyElementTool: ToolCallMessagePartComponent<
  StudyElementArgs,
  PresentedResult
> = ({ args, status }) => {
  const aui = useAui();
  const canAnswer = useAuiState((state) => state.thread.capabilities.answerToolCall);
  const id = useId();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [flipped, setFlipped] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  if (status.type === "incomplete" && status.reason === "error") {
    return (
      <p className="study-tool-error" role="alert">
        The tutor couldn’t prepare this study element.
      </p>
    );
  }

  if (
    !args ||
    typeof args.question !== "string" ||
    !Array.isArray(args.options) ||
    args.options.some(
      (option) =>
        !option ||
        typeof option.id !== "string" ||
        typeof option.label !== "string",
    )
  ) {
    return <p className="study-tool-status" role="status">Preparing study element...</p>;
  }

  const options = args.options;
  const flashcard = args.kind === "flashcard" && options.length === 1;
  const multiSelect = args.kind === "multi_select";

  const toggleOption = (optionId: string) => {
    setSelectedIds((selected) => {
      if (multiSelect) {
        return selected.includes(optionId)
          ? selected.filter((item) => item !== optionId)
          : [...selected, optionId];
      }
      return selected.includes(optionId) ? [] : [optionId];
    });
    setSubmitError(null);
  };

  const submitAnswer = async () => {
    if (!selectedIds.length || submitting || submitted || !canAnswer) return;

    setSubmitting(true);
    setSubmitError(null);
    const optionSummary = options
      .map((option) => `${option.id}: ${option.label}`)
      .join("; ");
    const text = [
      `Question type: ${args.kind}`,
      `Study question: ${args.question}`,
      `Options: ${optionSummary}`,
      `Selected option IDs: ${selectedIds.join(", ")}`,
    ].join("\n");

    try {
      await aui.thread.append({
        role: "user",
        content: [{ type: "text", text }],
      });
      setSubmitted(true);
    } catch (error) {
      setSubmitError(
        error instanceof Error ? error.message : "Could not submit your answer.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  if (flashcard) {
    return (
      <section className="study-element">
        <button
          type="button"
          className={`study-flashcard${flipped ? " is-flipped" : ""}`}
          aria-label={flipped ? "Show flashcard question" : "Reveal flashcard answer"}
          aria-pressed={flipped}
          onClick={() => setFlipped((value) => !value)}
        >
          <span className="study-flashcard-inner">
            <span className="study-flashcard-face" aria-hidden={flipped}>
              <span className="study-element-kicker">Flashcard - Question</span>
              <span>{args.question}</span>
              <span className="study-element-hint">Select to reveal the answer</span>
            </span>
            <span className="study-flashcard-face study-flashcard-back" aria-hidden={!flipped}>
              <span className="study-element-kicker">Answer</span>
              <span>{options[0].label}</span>
              <span className="study-element-hint">Select to see the question</span>
            </span>
          </span>
        </button>
      </section>
    );
  }

  return (
    <section className="study-element">
      <fieldset className="study-question">
        <legend>{args.question}</legend>
        {options.map((option, index) => {
          const optionId = `${id}-${index}`;
          return (
            <label className="study-option" key={option.id} htmlFor={optionId}>
              <input
                id={optionId}
                type={multiSelect ? "checkbox" : "radio"}
                name={id}
                value={option.id}
                checked={selectedIds.includes(option.id)}
                disabled={!canAnswer || submitted || submitting}
                onChange={() => toggleOption(option.id)}
              />
              <span>{option.label}</span>
            </label>
          );
        })}
      </fieldset>
      {canAnswer ? (
        <button
          type="button"
          className="study-submit"
          disabled={!selectedIds.length || submitted || submitting}
          onClick={() => void submitAnswer()}
        >
          {submitted ? "Answer submitted" : submitting ? "Submitting…" : "Submit answer"}
        </button>
      ) : (
        <p className="study-tool-status">This conversation is read-only.</p>
      )}
      {submitError && <p className="study-tool-error" role="alert">{submitError}</p>}
    </section>
  );
};

const HtmlPreviewTool: ToolCallMessagePartComponent<
  HtmlPreviewArgs,
  PresentedResult
> = ({ args, status }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const title = typeof args?.title === "string" ? args.title : "";
  const html =
    typeof args?.html === "string" &&
    args.html.length > 0 &&
    args.html.length <= MAX_PREVIEW_HTML_LENGTH
      ? args.html
      : "";

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !html) return;

    const controller = new AbortController();
    let activeFrame: Awaited<ReturnType<typeof htmlFrame.renderHtml>> | undefined;
    let active = true;
    setError(null);

    htmlFrame
      .renderHtml(html, container, { signal: controller.signal })
      .then(async (frame) => {
        if (!active) {
          frame.dispose();
          return;
        }
        activeFrame = frame;
        frame.iframe.title = title;
        await frame.fullyLoadedPromiseWithTimeout(10_000);
      })
      .catch((cause: unknown) => {
        if (!active || controller.signal.aborted) return;
        setError(
          cause instanceof Error ? cause.message : "The isolated preview could not load.",
        );
      });

    return () => {
      active = false;
      controller.abort();
      activeFrame?.dispose();
    };
  }, [html, title]);

  if (status.type === "incomplete" && status.reason === "error") {
    return (
      <p className="study-tool-error" role="alert">
        The tutor couldn’t prepare the HTML preview.
      </p>
    );
  }

  if (
    !args ||
    !title.trim() ||
    !html
  ) {
    return <p className="study-tool-error" role="alert">This preview is invalid or too large.</p>;
  }

  return (
    <section className="study-element html-preview">
      <h3>{args.title}</h3>
      <div
        ref={containerRef}
        className="html-preview-frame"
        aria-label={`${args.title} isolated HTML preview`}
      />
      {error && <p className="study-tool-error" role="alert">{error}</p>}
    </section>
  );
};

export const backendToolToolkit = defineToolkit({
  present_study_element: {
    type: "backend",
    display: "standalone",
    render: StudyElementTool,
  },
  present_html_preview: {
    type: "backend",
    display: "standalone",
    render: HtmlPreviewTool,
  },
});
