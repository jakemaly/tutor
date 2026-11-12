import type { SyntaxHighlighterProps } from "@assistant-ui/react-markdown";
import { useAuiState } from "@assistant-ui/react";
import { useShikiHighlighter } from "react-shiki";

export const SyntaxHighlighter = ({ code, language }: SyntaxHighlighterProps) => {
  const isStreaming = useAuiState((state) => state.part.status.type === "running");
  const trimmed = code.trim();
  if (isStreaming) {
    return (
      <div className="aui-shiki-base aui-shiki-streaming">
        <pre>
          <code>{trimmed}</code>
        </pre>
      </div>
    );
  }

  return <SettledCode code={trimmed} language={language || "text"} />;
};

const SettledCode = ({ code, language }: { code: string; language: string }) => {
  const highlighted = useShikiHighlighter(
    code,
    language,
    "github-dark-default",
    { delay: 150 },
  );

  return (
    <div className="aui-shiki-base">
      {highlighted ?? (
        <pre>
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
};
