import type { Editor, TLShapeId } from "tldraw";

// Matches the attachment limit enforced by attachments.ts and the backend.
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

export type WhiteboardSnapshot = {
  file: File;
  texts: string[];
};

/** Export the selected shapes as a PNG plus any plain text they contain. */
export async function exportSelection(
  editor: Editor,
  ids: TLShapeId[],
): Promise<WhiteboardSnapshot> {
  let blob: Blob | undefined;
  for (const scale of [2, 1, 0.5]) {
    ({ blob } = await editor.toImage(ids, {
      format: "png",
      background: true,
      padding: 16,
      scale,
    }));
    if (blob.size <= MAX_IMAGE_BYTES) break;
  }
  if (!blob || blob.size > MAX_IMAGE_BYTES) {
    throw new Error("The selection is too large to send. Select a smaller area.");
  }

  const texts: string[] = [];
  for (const id of editor.getShapeAndDescendantIds(ids)) {
    const shape = editor.getShape(id);
    if (!shape) continue;
    const text = editor.getShapeUtil(shape).getText(shape)?.trim();
    if (text) texts.push(text);
  }

  return {
    file: new File([blob], "whiteboard-selection.png", { type: "image/png" }),
    texts,
  };
}

/** Combine the learner's question with the text read from the board. */
export function buildWhiteboardPrompt(prompt: string, texts: string[]): string {
  const question = prompt.trim() || "Can you help me with this part of my whiteboard?";
  if (!texts.length) return question;
  return `${question}\n\nText on the selected board area:\n${texts
    .map((text) => `- ${text.replace(/\s*\n\s*/g, " ")}`)
    .join("\n")}`;
}
