import { createAssistantStream } from "assistant-stream";
import {
  createSimpleTitleAdapter,
  InMemoryThreadListAdapter,
  type ThreadHistoryAdapter,
} from "@assistant-ui/react";

type ThreadRepository = Awaited<ReturnType<ThreadHistoryAdapter["load"]>>;
type ThreadRepositoryItem = Parameters<ThreadHistoryAdapter["append"]>[0];
type ThreadTitleMessages = Parameters<
  ReturnType<typeof createSimpleTitleAdapter>["generateTitle"]
>[0];

const threadHistories = new Map<string, ThreadRepository>();
const titleGenerator = createSimpleTitleAdapter();

export function clearSessionThreadHistory(threadId: string) {
  threadHistories.delete(threadId);
}

export class SessionThreadHistoryAdapter implements ThreadHistoryAdapter {
  constructor(private readonly getThreadId: () => string | undefined) {}

  async load(): Promise<ThreadRepository> {
    const threadId = this.getThreadId();
    const repository = threadId ? threadHistories.get(threadId) : undefined;

    return repository
      ? { ...repository, messages: [...repository.messages] }
      : { messages: [] };
  }

  async append(item: ThreadRepositoryItem): Promise<void> {
    this.upsert(item, true);
  }

  async update(item: ThreadRepositoryItem): Promise<void> {
    this.upsert(item, false);
  }

  async delete(items: ThreadRepositoryItem[]): Promise<void> {
    const threadId = this.requireThreadId();
    const repository = threadHistories.get(threadId);
    if (!repository) return;

    const deletedIds = new Set(items.map(({ message }) => message.id));
    const messages = repository.messages.filter(
      ({ message }) => !deletedIds.has(message.id),
    );
    const headId =
      repository.headId != null && deletedIds.has(repository.headId)
        ? null
        : repository.headId;

    threadHistories.set(threadId, { ...repository, messages, headId });
  }

  private upsert(item: ThreadRepositoryItem, updateHead: boolean) {
    const threadId = this.requireThreadId();
    const repository = threadHistories.get(threadId) ?? { messages: [] };
    const index = repository.messages.findIndex(
      ({ message }) => message.id === item.message.id,
    );
    const messages = [...repository.messages];

    if (index === -1) messages.push(item);
    else messages[index] = item;

    threadHistories.set(threadId, {
      ...repository,
      messages,
      ...(updateHead || repository.headId == null
        ? { headId: item.message.id }
        : {}),
    });
  }

  private requireThreadId() {
    const threadId = this.getThreadId();
    if (!threadId) {
      throw new Error("Cannot save conversation history before thread initialization.");
    }
    return threadId;
  }
}

export class SessionThreadListAdapter extends InMemoryThreadListAdapter {
  override async generateTitle(
    _remoteId?: string,
    messages: ThreadTitleMessages = [],
  ) {
    const title = await titleGenerator.generateTitle(messages);
    return createAssistantStream((controller) => {
      controller.appendText(title);
    });
  }

  override async delete(remoteId?: string): Promise<void> {
    if (remoteId === undefined) {
      await super.delete();
      return;
    }

    clearSessionThreadHistory(remoteId);
    await super.delete(remoteId);
  }
}
