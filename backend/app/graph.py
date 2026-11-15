"""Small, provider-independent LangGraph chat graph."""

# LangChain supplies the provider-neutral model and messages; LangGraph defines the workflow.
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from openai import APIError

from app.artifacts import ArtifactBackend, ArtifactStore
from app.skills import build_system_prompt, discover_skills, make_load_skill
from app.study_document import make_create_study_document

SKILLS = discover_skills()
SYSTEM_PROMPT = build_system_prompt(SKILLS)

# Some providers (for example Groq) check tool arguments against the schema themselves and abort
# the response, so the model never gets LangGraph's usual "fix this and call again" feedback.
MAX_TOOL_CALL_RETRIES = 2
MAX_FEEDBACK_CHARS = 600


def is_tool_call_rejection(error: BaseException) -> bool:
    """True when the provider refused the model's tool call as malformed or off-schema."""
    return isinstance(error, APIError) and (
        getattr(error, "code", None) == "tool_use_failed"
        or "tool call validation failed" in str(error).lower()
    )


def create_graph(model: BaseChatModel, artifacts: ArtifactBackend | None = None):
    """Let the model load relevant skills and create documents, then stream its answer."""
    artifacts = artifacts if artifacts is not None else ArtifactStore()
    tools = [make_load_skill(SKILLS), make_create_study_document(artifacts)]
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
        messages = list(state["messages"])
        for attempt in range(MAX_TOOL_CALL_RETRIES + 1):
            # Stream the chain so the API can forward tokens before the full reply is ready.
            response = None
            try:
                async for chunk in chat_chain.astream({"messages": messages}):
                    # LangGraph stores one final assistant message; merge each delta into it.
                    response = chunk if response is None else response + chunk
            except APIError as error:
                if attempt == MAX_TOOL_CALL_RETRIES or not is_tool_call_rejection(error):
                    raise
                # Retry with the provider's complaint, for this attempt only: it is not saved
                # in the conversation state.
                messages = [
                    *state["messages"],
                    HumanMessage(
                        content=(
                            f"Your last tool call was rejected: {str(error)[:MAX_FEEDBACK_CHARS]}\n"
                            "Call the tool again. Every argument is required, and nested "
                            "arguments such as `guide` must be JSON objects, never strings."
                        )
                    ),
                ]
                continue
            return {"messages": [response] if response is not None else []}
        raise AssertionError("unreachable: the last attempt returns or raises")

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    # Bound repeated tool requests if a model fails to move on to its answer.
    return graph.compile().with_config({"recursion_limit": 12})
