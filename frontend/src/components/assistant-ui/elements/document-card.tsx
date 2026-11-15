import { useEffect, useState, type FC } from "react";
import type { ToolCallMessagePartProps } from "@assistant-ui/react";
import { DownloadIcon, EyeIcon, EyeOffIcon, FileTextIcon } from "lucide-react";

// Metadata only: the backend never sends file bytes through the chat stream.
type DocumentResult = {
  url: string;
  filename: string;
  format: "pdf" | "docx";
  title: string;
  size: number;
  expires_at: number; // Unix seconds
  outline: { heading: string; blocks: number }[];
  error?: string;
};

const EXPIRED_MESSAGE = "This file has expired. Ask me to create it again.";
const FAILED_MESSAGE = "The file could not be loaded. Check your connection and try again.";

class ExpiredError extends Error {
  constructor() {
    super(EXPIRED_MESSAGE);
  }
}

const formatSize = (bytes: number) =>
  bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

// Fetch the file ourselves so an expired link shows a clear message instead of a raw JSON error.
async function fetchFile(url: string): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw new Error(FAILED_MESSAGE);
  }
  if (response.status === 404) throw new ExpiredError();
  if (!response.ok) throw new Error(FAILED_MESSAGE);
  return response.blob();
}

export const DocumentCard: FC<ToolCallMessagePartProps<{ format?: string }, DocumentResult>> = ({
  args, result, status,
}) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState<"preview" | "download" | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const expiresAt = result?.expires_at;
  const [expired, setExpired] = useState(
    () => expiresAt !== undefined && Date.now() / 1000 > expiresAt,
  );

  // Switch to "expired" at the moment the server drops the file, without needing a click.
  useEffect(() => {
    if (expiresAt === undefined) return;
    const timer = window.setTimeout(() => setExpired(true), Math.max(expiresAt * 1000 - Date.now(), 0));
    return () => window.clearTimeout(timer);
  }, [expiresAt]);

  // Release the in-memory PDF when the preview closes or the card goes away.
  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl],
  );

  const format = (result?.format ?? args.format ?? "pdf").toUpperCase();

  if (result?.error || status.type === "incomplete") {
    return (
      <p role="alert" className="my-2 text-sm text-muted-foreground">
        The {format} could not be created. Ask again, or say what to include.
      </p>
    );
  }
  if (!result) {
    return (
      <p role="status" className="my-2 text-sm text-muted-foreground">
        Creating {format}…
      </p>
    );
  }

  const run = async (kind: "preview" | "download", task: () => Promise<void>) => {
    setBusy(kind);
    setProblem(null);
    try {
      await task();
    } catch (error) {
      if (error instanceof ExpiredError) setExpired(true);
      setProblem(error instanceof Error ? error.message : FAILED_MESSAGE);
    } finally {
      setBusy(null);
    }
  };

  const togglePreview = () => {
    if (previewUrl) {
      setPreviewUrl(null);
      return;
    }
    void run("preview", async () => {
      setPreviewUrl(URL.createObjectURL(await fetchFile(`${result.url}?inline=true`)));
    });
  };

  const download = () =>
    void run("download", async () => {
      const href = URL.createObjectURL(await fetchFile(result.url));
      const link = document.createElement("a");
      link.href = href;
      link.download = result.filename;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(href), 30_000);
    });

  const showPreview = previewUrl !== null && !expired;
  return (
    <div className="border-foreground/10 bg-muted/30 my-3 overflow-hidden rounded-xl border">
      <div className="flex items-center gap-3 p-3">
        <FileTextIcon className="text-muted-foreground size-8 shrink-0" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium">{result.title}</p>
          <p className="text-muted-foreground text-sm">
            {result.format.toUpperCase()} · {formatSize(result.size)} ·{" "}
            {expired ? "expired, ask me to create it again" : "available for 30 minutes"}
          </p>
        </div>
        {!expired && (
          <div className="flex shrink-0 items-center gap-1.5">
            {result.format === "pdf" && (
              <button
                type="button"
                onClick={togglePreview}
                disabled={busy !== null}
                aria-expanded={showPreview}
                className="hover:bg-accent inline-flex h-8 items-center gap-1.5 rounded-lg border px-3 text-sm disabled:opacity-60"
              >
                {showPreview ? <EyeOffIcon className="size-4" /> : <EyeIcon className="size-4" />}
                {busy === "preview" ? "Loading…" : showPreview ? "Hide preview" : "Preview"}
              </button>
            )}
            <button
              type="button"
              onClick={download}
              disabled={busy !== null}
              className="bg-primary text-primary-foreground inline-flex h-8 items-center gap-1.5 rounded-lg px-3 text-sm disabled:opacity-60"
            >
              <DownloadIcon className="size-4" aria-hidden="true" />
              {busy === "download" ? "Preparing…" : "Download"}
            </button>
          </div>
        )}
      </div>
      {problem && (
        <p role="alert" className="text-muted-foreground border-t px-3 py-2 text-sm">
          {problem}
        </p>
      )}
      {showPreview && (
        <iframe
          title={`Preview of ${result.title}`}
          src={previewUrl}
          className="h-[28rem] w-full border-t bg-white"
        />
      )}
      {!showPreview && result.outline.length > 0 && (
        <ol className="text-muted-foreground list-inside list-decimal border-t px-3 py-2 text-sm">
          {result.outline.map((section, index) => (
            // Headings are written by the model and can repeat, so the index keeps keys unique.
            <li key={`${index}-${section.heading}`}>{section.heading}</li>
          ))}
        </ol>
      )}
    </div>
  );
};
