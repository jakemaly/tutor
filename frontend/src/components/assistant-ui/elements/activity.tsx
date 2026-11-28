import { useEffect, useState, type FC, type ReactNode } from "react";
import { useAuiState, type ToolCallMessagePartProps } from "@assistant-ui/react";
import { ThinkingIndicator } from "./thinking-indicator";
import { TypingIndicator } from "./typing-indicator";
import { ToolCall } from "./tool-call";
import { ToolTimeline } from "./tool-timeline";
import { WebSearch } from "./web-search";

type ModelStatus = {
  stage: string;
  detail?: string;
  gpu?: string;
  gpu_memory_mib?: number;
  provider?: string;
  region?: string;
  weights_source?: string;
  quantization?: string;
  service?: { queued: number; containers: number; running: number };
  events?: { label: string; at: string | null }[];
};
const LABELS: Record<string, string> = {
  waiting: "Waiting for model", starting: "Starting GPU", loading: "Loading model",
  warming: "Warming up model", creating_snapshot: "Creating snapshot",
  restoring: "Restoring snapshot", ready: "Preparing answer", failed: "Model startup failed",
};

export const AssistantWaiting: FC = () => {
  const running = useAuiState((s) => s.message.status?.type === "running");
  const messageId = useAuiState((s) => s.message.id);
  const parts = useAuiState((s) => s.message.parts);
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const last = parts[parts.length - 1];
  const pendingTool = [...parts].reverse().find((part) => part.type === "tool-call" && part.result === undefined);
  const waiting = running && (!last || last.type !== "text");

  const observeStartup = waiting && !last;
  useEffect(() => {
    if (!observeStartup) return;
    setStatus(null);
    setElapsed(0);
    const started = performance.now();
    const abort = new AbortController();
    let timer: number | undefined;
    const update = async () => {
      const seconds = Math.floor((performance.now() - started) / 1000);
      try {
        const response = await fetch(`/api/model-status?message_id=${encodeURIComponent(messageId)}`, { signal: abort.signal });
        if (response.ok && !abort.signal.aborted) setStatus(await response.json() as ModelStatus);
      } catch { /* Model telemetry is optional and never blocks chat. */ }
      if (!abort.signal.aborted) {
        setElapsed(seconds);
        timer = window.setTimeout(() => void update(), 3000);
      }
    };
    void update();
    return () => { abort.abort(); window.clearTimeout(timer); };
  }, [messageId, observeStartup]);

  if (!waiting) return null;
  if (last?.type === "tool-call" && !pendingTool) return <TypingIndicator variant="bare" className="my-2" />;
  const toolLabel = pendingTool?.type === "tool-call"
    ? pendingTool.toolName === "web_search" ? "Searching the web" : "Preparing teaching guidance"
    : null;
  const serviceLabel = status && LABELS[status.stage];
  const label = toolLabel ?? (serviceLabel ? `Model service: ${serviceLabel}` : "Thinking");
  const gpu = status?.gpu && status.gpu_memory_mib
    ? `${status.gpu} (${(status.gpu_memory_mib / 1024).toFixed(1)} GiB)` : status?.gpu;
  const location = [gpu, status?.provider, status?.region].filter(Boolean).join(" · ");
  return <div className="my-2" role="status" aria-live="polite">
    <ThinkingIndicator label={label} elapsed={elapsed > 0 ? `${elapsed}s` : undefined} />
    {!toolLabel && status && <div className="text-muted-foreground mt-2 space-y-1 border-l border-border pl-3 text-xs" data-slot="modal-telemetry">
      {status.detail ? <p className="text-foreground/80">{status.detail}</p> :
        <p>{status.stage === "unavailable" ? "Waiting for startup telemetry." : "No startup event received yet."}</p>}
      {location && <p>GPU: {location}</p>}
      {status.weights_source && <p>Weights: {status.weights_source}</p>}
      {status.quantization && <p>Quantization: {status.quantization}</p>}
      {status.service && <p>Modal service: {status.service.queued} queued · {status.service.running} running · {status.service.containers} {status.service.containers === 1 ? "container" : "containers"}</p>}
      {status.events && status.events.length > 0 && <ol aria-label="Recent startup events" className="space-y-0.5 pt-1">
        {status.events.slice(-4).map((event, index) => <li key={`${event.at}-${index}`}>
          {event.at && <time className="mr-2 tabular-nums" dateTime={event.at}>{new Date(event.at).toLocaleTimeString()}</time>}
          {event.label}
        </li>)}
      </ol>}
    </div>}
  </div>;
};


export function ActivityGroup({ groupKey, indices, children }: { groupKey: string | undefined; indices: number[]; children?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const parts = useAuiState((s) => s.message.parts);
  const visible = indices.filter((index) => {
    const part = parts[index];
    return part?.type === "tool-call" && !(part.result && typeof part.result === "object" && "skipped" in part.result && part.result.skipped);
  });
  if (!groupKey) return <>{children}</>;
  if (!visible.length) return null;
  const active = visible.some((index) => {
    const part = parts[index];
    return part?.type === "tool-call" && part.result === undefined && part.status?.type === "running";
  });
  return <ToolTimeline steps={[]} visibleSteps={0} stats={[]} streaming={active} open={open} onOpenChange={setOpen} activeLabel="Working" restingLabel={`${visible.length} ${visible.length === 1 ? "step" : "steps"}`} className="my-2 max-w-none">{children}</ToolTimeline>;
}

export const SkillToolStatus: FC<ToolCallMessagePartProps<{ name?: string }, { loaded: boolean }>> = ({ args, result, status, isError }) => {
  const [open, setOpen] = useState(false);
  const cancelled = status.type === "incomplete" && status.reason === "cancelled";
  const failed = !cancelled && (!!isError || result?.loaded === false || status.type === "incomplete");
  return <ToolCall cancelled={cancelled} label={cancelled ? "Guidance cancelled" : failed ? "Guidance unavailable" : "Guidance ready"} activeLabel="Preparing guidance" query={args.name || ""} request={args.name || ""} result={cancelled ? "Cancelled." : failed ? "Could not load guidance." : "Teaching guidance prepared."} running={status.type === "running"} failed={failed} open={open} onOpenChange={setOpen} />;
};

type SearchResult = { title: string; url: string };
export const WebSearchToolStatus: FC<ToolCallMessagePartProps<{ query?: string }, { results: SearchResult[]; skipped?: boolean }>> = ({ args, result, status, isError }) => {
  if (result?.skipped) return null;
  const cancelled = status.type === "incomplete" && status.reason === "cancelled";
  const failed = !cancelled && (!!isError || status.type === "incomplete");
  const results = (result?.results ?? []).flatMap((item) => {
    try {
      const url = new URL(item.url);
      if (!["http:", "https:"].includes(url.protocol)) return [];
      return [{ title: item.title, domain: url.hostname, url: url.href }];
    } catch { return []; }
  });
  return <div className="my-1">{cancelled ? <p className="text-muted-foreground text-sm">Search cancelled</p> : failed ? <p className="text-destructive text-sm">Search failed</p> : <WebSearch query={args.query || ""} results={results} visibleResults={results.length} searching={!result} cycle={0} />}</div>;
};
