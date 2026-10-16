"""Small, provider-independent LangGraph chat graph."""

# LangChain supplies the provider-neutral model and messages; LangGraph defines the workflow.
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, MessagesState, StateGraph

SYSTEM_PROMPT = "You are a helpful, clear assistant. Answer questions directly and accurately."


def create_graph(model: BaseChatModel):
    """Compile a single-node graph around any LangChain chat model."""
    # LCEL formats the prompt, then passes its messages to the configured model.
    chat_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("messages"),
            ]
        )
        | model
    )

    async def chat(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Stream one assistant reply and return it as the graph's new message."""
        # Stream the chain so the API can forward tokens before the full reply is ready.
        response = None
        async for chunk in chat_chain.astream({"messages": state["messages"]}):
            # LangGraph stores one final assistant message; merge each streamed delta into it.
            response = chunk if response is None else response + chunk
        return {"messages": [response] if response is not None else []}

    # One node is enough for this starter; later workflow steps can be added as graph nodes.
    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()
