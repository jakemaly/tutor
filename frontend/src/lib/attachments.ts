import type {
  Attachment,
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";

type UploadAttachment = PendingAttachment & { sourceId?: string };

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = new Set([
  "application/pdf",
  "image/jpeg",
  "image/png",
  "image/webp",
]);
const activeUploads = new Map<string, XMLHttpRequest>();

async function* uploadFile(file: File, id: string): AsyncGenerator<number> {
  const request = new XMLHttpRequest();
  activeUploads.set(id, request);
  let progress = 0;
  let lastYielded = 0;
  let finished = false;
  let failure: Error | undefined;
  let wake = () => {};
  const notify = () => wake();

  request.open("PUT", `/api/attachments/${id}`);
  request.setRequestHeader("Content-Type", file.type);
  request.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
  request.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      progress = Math.min(99, Math.round((event.loaded / event.total) * 100));
      notify();
    }
  };
  request.onload = () => {
    activeUploads.delete(id);
    if (request.status >= 200 && request.status < 300) {
      progress = 100;
    } else {
      let message = "Upload failed. Please try again.";
      try {
        message = JSON.parse(request.responseText).detail || message;
      } catch {
        // Keep the generic message when the server does not return JSON.
      }
      failure = new Error(message);
    }
    finished = true;
    notify();
  };
  request.onerror = () => {
    activeUploads.delete(id);
    failure = new Error("Upload failed. Check your connection.");
    finished = true;
    notify();
  };
  request.onabort = () => {
    activeUploads.delete(id);
    failure = new Error("Upload was cancelled.");
    finished = true;
    notify();
  };
  request.send(file);

  while (!finished || progress > lastYielded) {
    if (progress > lastYielded) {
      lastYielded = progress;
      yield progress;
      continue;
    }
    if (!finished) await new Promise<void>((resolve) => (wake = resolve));
  }
  if (failure) throw failure;
}

export const attachmentAdapter: AttachmentAdapter = {
  accept: "application/pdf,image/jpeg,image/png,image/webp",

  async *add({ file }) {
    const id = crypto.randomUUID();
    const type = file.type.startsWith("image/") ? "image" : "document";
    const base: UploadAttachment = {
      id,
      type,
      name: file.name,
      contentType: file.type,
      file,
      status: { type: "running", reason: "uploading", progress: 0 },
    };

    if (!ACCEPTED_TYPES.has(file.type)) {
      yield {
        ...base,
        status: {
          type: "incomplete",
          reason: "error",
          message: "Choose a PDF, JPEG, PNG, or WebP file.",
        },
      };
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      yield {
        ...base,
        status: {
          type: "incomplete",
          reason: "error",
          message: "Attachments must be 10 MB or smaller.",
        },
      };
      return;
    }

    yield base;
    try {
      for await (const progress of uploadFile(file, id)) {
        yield { ...base, status: { type: "running", reason: "uploading", progress } };
      }
      yield {
        ...base,
        status: { type: "requires-action", reason: "composer-send" },
      };
    } catch (error) {
      yield {
        ...base,
        status: {
          type: "incomplete",
          reason: "error",
          message: error instanceof Error ? error.message : "Upload failed. Please try again.",
        },
      };
    }
  },

  async send(attachment): Promise<CompleteAttachment> {
    const uploaded = attachment as UploadAttachment;
    return {
      ...uploaded,
      status: { type: "complete" },
      content: [
        {
          type: "file",
          data: uploaded.sourceId ?? uploaded.id,
          mimeType: uploaded.contentType ?? "application/octet-stream",
          filename: uploaded.name,
          sourceType: "id",
        },
      ],
    };
  },

  async remove(attachment: Attachment) {
    activeUploads.get(attachment.id)?.abort();
    const response = await fetch(`/api/attachments/${encodeURIComponent(attachment.id)}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new Error("Could not remove the uploaded attachment.");
  },
};
