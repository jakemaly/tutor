"""Small, provider-independent LangGraph chat graph."""

# LangChain supplies the provider-neutral model and messages; LangGraph defines the workflow.
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.skills import build_system_prompt, discover_skills, make_load_skill

SKILLS = discover_skills()
SYSTEM_PROMPT = (
    build_system_prompt(SKILLS)
    + """

When attached sources are provided, cite factual claims with the exact marker
[[cite:SOURCE_ID:PAGE_NUMBER]], using only source IDs and page numbers shown in the
attachment context. Never invent citations. For image attachments, cite page 1. Do not
emit citation markers when no attached source supports the claim."""
)


def create_graph(model: BaseChatModel):
    """Let the model load relevant skills, then stream its answer."""
    tools = [make_load_skill(SKILLS)]
    # LCEL formats the prompt, then passes its messages to the configured model.
    chat_chain = ChatPromptTemplate.from_messages(
        [
            # Skills can contain JSON/code braces; they are literal text, not templates.
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder("messages"),
        ]
    ) | model.bind_tools(tools)

    async def chat(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Stream one assistant reply and return it as the graph's new message."""
        # Stream the chain so the API can forward tokens before the full reply is ready.
        response = None
        async for chunk in chat_chain.astream({"messages": state["messages"]}):
            # LangGraph stores one final assistant message; merge each streamed delta into it.
            response = chunk if response is None else response + chunk
        return {"messages": [response] if response is not None else []}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    # Bound repeated tool requests if a model fails to move on to its answer.
    return graph.compile().with_config({"recursion_limit": 12})
