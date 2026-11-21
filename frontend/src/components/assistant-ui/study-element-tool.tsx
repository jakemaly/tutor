import {
  defineToolkit,
  type ToolCallMessagePartComponent,
} from "@assistant-ui/react";
import { SafeContentFrame } from "safe-content-frame";
import { useEffect, useId, useRef, useState, type FC } from "react";

type StudyOption = {
  id: string;
  label: string;
};

type StudyElement = {
  kind: "flashcard" | "mcq" | "true_false" | "multi_select";
  question: string;
  options: StudyOption[];
  correct_option_ids?: string[];
  explanation?: string | null;
};

type StudySetArgs = {
  title: string;
  elements: StudyElement[];
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

function isStudyElement(value: unknown): value is StudyElement {
  if (!value || typeof value !== "object") return false;
  const element = value as Partial<StudyElement>;
  if (
    !["flashcard", "mcq", "true_false", "multi_select"].includes(
      element.kind ?? "",
    ) ||
    typeof element.question !== "string" ||
    !element.question.trim() ||
    !Array.isArray(element.options) ||
    element.options.some(
      (option) =>
        !option ||
        typeof option.id !== "string" ||
        typeof option.label !== "string",
    )
  ) {
    return false;
  }
  if (element.kind === "flashcard") return element.options.length === 1;

  const correctIds = element.correct_option_ids;
  if (!Array.isArray(correctIds) || correctIds.some((id) => typeof id !== "string")) {
    return false;
  }
  const optionIds = element.options.map((option) => option.id);
  if (
    optionIds.length < 2 ||
    correctIds.length === 0 ||
    correctIds.some((id) => !optionIds.includes(id))
  ) {
    return false;
  }
  if (element.kind === "true_false") {
    return (
      optionIds.length === 2 &&
      new Set(element.options.map((option) => option.label.toLowerCase())).size === 2 &&
      element.options.some((option) => option.label.toLowerCase() === "true") &&
      element.options.some((option) => option.label.toLowerCase() === "false") &&
      correctIds.length === 1
    );
  }
  if (element.kind === "mcq") return correctIds.length === 1;
  return true;
}

function isStudySet(value: unknown): value is StudySetArgs {
  if (!value || typeof value !== "object") return false;
  const set = value as Partial<StudySetArgs>;
  return (
    typeof set.title === "string" &&
    Boolean(set.title.trim()) &&
    Array.isArray(set.elements) &&
    set.elements.length >= 2 &&
    set.elements.every(isStudyElement)
  );
}

const StudyItem: FC<{ element: StudyElement }> = ({ element }) => {
  const id = useId();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [flipped, setFlipped] = useState(false);
  const [graded, setGraded] = useState<boolean | null>(null);
  const isMultiSelect = element.kind === "multi_select";
  const correctIds = element.correct_option_ids ?? [];

  const toggleOption = (optionId: string) => {
    setSelectedIds((selected) => {
      if (isMultiSelect) {
        return selected.includes(optionId)
          ? selected.filter((item) => item !== optionId)
          : [...selected, optionId];
      }
      return selected.includes(optionId) ? [] : [optionId];
    });
  };

  const submitAnswer = () => {
    if (!selectedIds.length || graded !== null) return;
    const selected = new Set(selectedIds);
    setGraded(
      selected.size === correctIds.length &&
        correctIds.every((optionId) => selected.has(optionId)),
    );
  };

  if (element.kind === "flashcard") {
    return (
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
            <span>{element.question}</span>
            <span className="study-element-hint">Select to reveal the answer</span>
          </span>
          <span className="study-flashcard-face study-flashcard-back" aria-hidden={!flipped}>
            <span className="study-element-kicker">Answer</span>
            <span>{element.options[0].label}</span>
            <span className="study-element-hint">Select to see the question</span>
          </span>
        </span>
      </button>
    );
  }

  return (
    <div className="study-item">
      <fieldset className="study-question">
        <legend>{element.question}</legend>
        {isMultiSelect && (
          <span className="study-element-hint">Select all that apply</span>
        )}
        {element.options.map((option, index) => {
          const optionId = `${id}-${index}`;
          return (
            <label className="study-option" key={option.id} htmlFor={optionId}>
              <input
                id={optionId}
                type={isMultiSelect ? "checkbox" : "radio"}
                name={id}
                value={option.id}
                checked={selectedIds.includes(option.id)}
                disabled={graded !== null}
                onChange={() => toggleOption(option.id)}
              />
              <span>{option.label}</span>
            </label>
          );
        })}
      </fieldset>
      <button
        type="button"
        className="study-submit"
        disabled={!selectedIds.length || graded !== null}
        onClick={submitAnswer}
      >
        {graded === null ? "Check answer" : "Answer checked"}
      </button>
      {graded !== null && (
        <div
          className={`study-feedback${graded ? " is-correct" : " is-incorrect"}`}
          role="status"
        >
          <strong>{graded ? "Correct!" : "Not quite."}</strong>
          {!graded && (
            <p>
              Correct answer:{" "}
              {element.options
                .filter((option) => correctIds.includes(option.id))
                .map((option) => option.label)
                .join(", ")}
            </p>
          )}
          {element.explanation && <p>{element.explanation}</p>}
        </div>
      )}
    </div>
  );
};

const StudyElementTool: ToolCallMessagePartComponent<
  StudyElement,
  PresentedResult
> = ({ args, status }) => {
  if (status.type === "incomplete" && status.reason === "error") {
    return (
      <p className="study-tool-error" role="alert">
        The tutor couldn’t prepare this study element.
      </p>
    );
  }
  if (!isStudyElement(args)) {
    return <p className="study-tool-error" role="alert">This study element is invalid.</p>;
  }

  return (
    <section className="study-element">
      <StudyItem element={args} />
    </section>
  );
};

const StudySetTool: ToolCallMessagePartComponent<
  StudySetArgs,
  PresentedResult
> = ({ args, status }) => {
  const [activeIndex, setActiveIndex] = useState(0);

  if (status.type === "incomplete" && status.reason === "error") {
    return (
      <p className="study-tool-error" role="alert">
        The tutor couldn’t prepare this study set.
      </p>
    );
  }
  if (!isStudySet(args)) {
    return <p className="study-tool-error" role="alert">This study set is invalid.</p>;
  }

  return (
    <section className="study-element study-set">
      <header className="study-set-header">
        <h3>{args.title}</h3>
        <span aria-live="polite">
          Card {activeIndex + 1} of {args.elements.length}
        </span>
      </header>
      {args.elements.map((element, index) => (
        <div key={`${index}-${element.question}`} hidden={activeIndex !== index}>
          <StudyItem element={element} />
        </div>
      ))}
      <nav className="study-set-navigation" aria-label="Study set cards">
        <button
          type="button"
          disabled={activeIndex === 0}
          onClick={() => setActiveIndex((value) => value - 1)}
        >
          Previous
        </button>
        <button
          type="button"
          disabled={activeIndex === args.elements.length - 1}
          onClick={() => setActiveIndex((value) => value + 1)}
        >
          Next
        </button>
      </nav>
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

  if (!title.trim() || !html) {
    return <p className="study-tool-error" role="alert">This preview is invalid or too large.</p>;
  }

  return (
    <section className="study-element html-preview">
      <h3>{title}</h3>
      <div
        ref={containerRef}
        className="html-preview-frame"
        aria-label={`${title} isolated HTML preview`}
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
  present_study_elements: {
    type: "backend",
    display: "standalone",
    render: StudySetTool,
  },
  present_html_preview: {
    type: "backend",
    display: "standalone",
    render: HtmlPreviewTool,
  },
});
