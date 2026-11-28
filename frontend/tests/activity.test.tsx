// @vitest-environment jsdom
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, act } from "@testing-library/react";
import { groupAdjacentTools } from "../src/components/assistant-ui/elements/group-activity";

const state = vi.hoisted(() => ({ message: { id: "message-1", status: { type: "running" }, parts: [] as unknown[] } }));
vi.mock("@assistant-ui/react", () => ({ useAuiState: (selector: (s: unknown) => unknown) => selector(state) }));
vi.mock("../src/components/assistant-ui/elements/tool-call", () => ({ ToolCall: ({ label }: { label: string }) => <div>{label}</div> }));
vi.mock("../src/components/assistant-ui/elements/thinking-indicator", () => ({ ThinkingIndicator: ({ label }: { label: string }) => <div>{label}</div> }));
vi.mock("../src/components/assistant-ui/elements/typing-indicator", () => ({ TypingIndicator: () => <div>Typing</div> }));
vi.mock("../src/components/assistant-ui/elements/tool-timeline", () => ({ ToolTimeline: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }));
vi.mock("../src/components/assistant-ui/elements/web-search", () => ({ WebSearch: () => <div>Search results</div> }));

import { AssistantWaiting, SkillToolStatus, WebSearchToolStatus } from "../src/components/assistant-ui/elements/activity";

afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); state.message.parts = []; });

describe("assistant activity", () => {
  it("groups adjacent tools without moving text or merging separate groups", () => {
    expect(groupAdjacentTools([{ type: "text" }, { type: "tool-call" }, { type: "tool-call" }, { type: "text" }, { type: "tool-call" }])).toEqual([
      { groupKey: undefined, indices: [0] }, { groupKey: "tools", indices: [1, 2] },
      { groupKey: undefined, indices: [3] }, { groupKey: "tools", indices: [4] },
    ]);
  });

  it("labels cancelled tools as cancelled, not failed", () => {
    const cancelled = { type: "incomplete", reason: "cancelled" };
    render(<><SkillToolStatus {...({ args: { name: "example" }, status: cancelled } as never)} /><WebSearchToolStatus {...({ args: { query: "test" }, status: cancelled } as never)} /></>);
    expect(screen.getByText("Guidance cancelled")).toBeTruthy();
    expect(screen.getByText("Search cancelled")).toBeTruthy();
    expect(screen.queryByText("Search failed")).toBeNull();
  });

  it("shows observed Modal details and service counts while waiting", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        stage: "loading", detail: "Loading model weights: 2 of 4 shards",
        gpu: "NVIDIA L40S", gpu_memory_mib: 46068, provider: "AWS", region: "us-east",
        weights_source: "Cached on Modal Volume",
        service: { queued: 1, running: 0, containers: 1 },
        events: [{ label: "Starting the vLLM server", at: null }],
      }),
    }));
    render(<AssistantWaiting />);
    expect(await screen.findByText("Loading model weights: 2 of 4 shards")).toBeTruthy();
    expect(screen.getByText("GPU: NVIDIA L40S (45.0 GiB) · AWS · us-east")).toBeTruthy();
    expect(screen.getByText("Modal service: 1 queued · 0 running · 1 container")).toBeTruthy();
    expect(screen.getByText("Starting the vLLM server")).toBeTruthy();
  });

  it("does not overlap status requests and aborts polling when text starts", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(() => new Promise<Response>(() => {}));
    vi.stubGlobal("fetch", fetchMock);
    const { rerender } = render(<AssistantWaiting />);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => { vi.advanceTimersByTime(12000); });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const signal = fetchMock.mock.calls[0][1]?.signal as AbortSignal;
    state.message.parts = [{ type: "text", text: "Hello" }];
    rerender(<AssistantWaiting />);
    expect(signal.aborted).toBe(true);
  });
});
