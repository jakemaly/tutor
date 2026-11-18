"use client";

import { PreviewCard } from "@base-ui/react/preview-card";
import { ChevronDownIcon, FileTextIcon } from "lucide-react";
import { useState, type ComponentProps, type FC } from "react";
import { cn } from "@/lib/utils";

export type CitationSource = {
  id: string;
  page: number;
  title: string;
  snippet: string;
  url: string;
};

export const InlineCitation: FC<ComponentProps<"p">> = ({
  children,
  className,
  ...props
}) => (
  <p
    data-slot="inline-citation"
    className={cn("text-foreground/90 text-sm leading-relaxed", className)}
    {...props}
  >
    {children}
  </p>
);

export function Citation({
  index,
  source,
  className,
}: {
  index: number;
  source: CitationSource;
  className?: string;
}) {
  return (
    <PreviewCard.Root>
      <PreviewCard.Trigger
        delay={0}
        render={<button type="button" />}
        aria-label={`Source ${index + 1}: ${source.title}`}
        className={cn(
          "bg-foreground/[0.06] text-muted-foreground hover:text-foreground ms-0.5 inline-flex h-4 min-w-4 translate-y-[-2px] cursor-pointer items-center justify-center rounded px-1 align-middle font-mono text-[10px] tabular-nums",
          className,
        )}
      >
        {index + 1}
      </PreviewCard.Trigger>
      <PreviewCard.Portal>
        <PreviewCard.Positioner side="top" sideOffset={8}>
          <PreviewCard.Popup className="bg-popover text-popover-foreground z-50 w-64 rounded-lg border p-3 shadow-lg">
            <p className="text-sm font-medium">{source.title}</p>
            <p className="text-muted-foreground mt-1 text-xs leading-relaxed">
              {source.snippet}
            </p>
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary mt-3 inline-block text-xs underline underline-offset-2"
            >
              Open cited page
            </a>
          </PreviewCard.Popup>
        </PreviewCard.Positioner>
      </PreviewCard.Portal>
    </PreviewCard.Root>
  );
}

export function Sources({
  sources,
}: {
  sources: readonly CitationSource[];
}) {
  const [open, setOpen] = useState(false);

  if (sources.length === 0) return null;

  return (
    <section className="mt-3 border-t pt-2" aria-label="Cited sources">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="text-muted-foreground hover:text-foreground inline-flex items-center gap-1.5 text-xs"
      >
        <FileTextIcon className="size-3.5" aria-hidden="true" />
        Sources ({sources.length})
        <ChevronDownIcon
          className={cn("size-3.5 transition-transform", open && "rotate-180")}
          aria-hidden="true"
        />
      </button>
      {open && (
        <ul className="mt-2 flex flex-col divide-y">
          {sources.map((source) => (
            <li key={source.url} className="py-1.5 first:pt-0 last:pb-0">
              <a
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground hover:text-primary text-xs underline-offset-2 hover:underline"
              >
                {source.title}
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}