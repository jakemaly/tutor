import type { SyntaxHighlighterProps } from "@assistant-ui/react-markdown";
import { useShikiHighlighter } from "react-shiki";

export const SyntaxHighlighter = ({ code, language }: SyntaxHighlighterProps) => {
  const trimmed = code.trim();
  // delay waits for streamed text to settle before coloring it
  const highlighted = useShikiHighlighter(
    trimmed,
    language || "text",
    "github-dark-default",
    { delay: 150 },
  );

  return (
    <div className="aui-shiki-base">
      {highlighted ?? (
        <pre>
          <code>{trimmed}</code>
        </pre>
      )}
    </div>
  );
};