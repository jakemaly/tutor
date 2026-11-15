"use client";

import { useAuiState } from "@assistant-ui/react";
import { cn } from "@/lib/utils";
import { mono } from "./surfaces";

const dayFormat = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  year: "numeric",
});
const timeFormat = new Intl.DateTimeFormat(undefined, {
  hour: "numeric",
  minute: "2-digit",
});

export function MessageDaySeparator() {
  const messageId = useAuiState((state) => state.message.id);
  const createdAt = useAuiState((state) => state.message.createdAt);
  const messages = useAuiState((state) => state.thread.messages);
  const messageIndex = messages.findIndex((message) => message.id === messageId);
  const previousMessage = messages
    .slice(0, messageIndex)
    .reverse()
    .find((message) => message.role !== "system");

  if (
    messageIndex < 0 ||
    (previousMessage &&
      dayFormat.format(previousMessage.createdAt) === dayFormat.format(createdAt))
  ) {
    return null;
  }

  return (
    <div
      aria-hidden="true"
      className="aui-message-day-separator col-[1/-1] flex max-h-0 w-full items-center gap-2.5 overflow-hidden py-0 opacity-0 transition-all group-hover/message-root:max-h-8 group-hover/message-root:py-1 group-hover/message-root:opacity-100"
    >
      <span className="bg-foreground/[0.08] h-px flex-1" />
      <span className={cn(mono, "text-muted-foreground")}>
        {dayFormat.format(createdAt)}
      </span>
      <span className="bg-foreground/[0.08] h-px flex-1" />
    </div>
  );
}

export function MessageTimestamp({ align = "start" }: { align?: "start" | "end" }) {
  const createdAt = useAuiState((state) => state.message.createdAt);

  return (
    <div
      className={cn(
        "aui-message-timestamp text-muted-foreground absolute z-10 whitespace-nowrap tabular-nums opacity-0 transition-[opacity,translate] duration-200 group-hover/message-root:translate-x-0 group-hover/message-root:opacity-100 motion-reduce:transition-none",
        mono,
        align === "end"
          ? "top-auto bottom-0 left-full ml-2 translate-x-1 translate-y-1/2"
          : "right-full bottom-7.5 mr-2 -translate-x-1",
      )}
    >
      {timeFormat.format(createdAt)}
    </div>
  );
}
