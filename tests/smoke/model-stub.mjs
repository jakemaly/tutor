import { createServer } from "node:http";

const model = "tutor-smoke";

createServer(async (request, response) => {
  if (request.method === "GET" && request.url === "/health") {
    response.writeHead(200).end("ok");
    return;
  }
  if (request.method !== "POST" || request.url !== "/v1/chat/completions") {
    response.writeHead(404).end();
    return;
  }

  let completion;
  try {
    const parts = [];
    for await (const part of request) parts.push(part);
    completion = JSON.parse(Buffer.concat(parts));
  } catch {
    response.writeHead(400).end();
    return;
  }
  if (completion?.model !== model || completion.stream !== true) {
    response.writeHead(400).end();
    return;
  }

  response.writeHead(200, { "Content-Type": "text/event-stream" });
  const chunks = [
    { delta: { role: "assistant", content: "Try drawing a fraction bar." }, finish_reason: null },
    { delta: {}, finish_reason: "stop" },
  ];
  for (const chunk of chunks) {
    const event = {
      id: "chatcmpl-tutor-smoke",
      object: "chat.completion.chunk",
      created: 0,
      model,
      choices: [{ index: 0, ...chunk }],
    };
    response.write("data: " + JSON.stringify(event) + "\n\n");
  }
  response.end("data: [DONE]\n\n");
}).listen(8101, "127.0.0.1");
