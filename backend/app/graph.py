"""Small, provider-independent LangGraph chat graph."""

# LangChain supplies the provider-neutral model and messages; LangGraph defines the workflow.
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.models import discover_context_window
from app.skills import build_system_prompt, discover_skills, make_load_skill

SKILLS = discover_skills()
SYSTEM_PROMPT = build_system_prompt(SKILLS)


def create_graph(model: BaseChatModel):
    """Let the model load relevant skills, then stream its answer."""
    tools = [make_load_skill(SKILLS)]
    chat_model = model.bind_tools(tools)
    system_message = SystemMessage(content=SYSTEM_PROMPT)

    async def chat(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Stream one assistant reply, resuming a length-limited vLLM stream."""
        messages = [system_message, *state["messages"]]
        # Token usage shows whether a length stop left room in the context window.
        stream_options = {"stream_usage": True}
        prefix = ""
        first_id = None
        generated_tokens = 0
        context_window = None

        while True:
            response = None
            async for chunk in chat_model.astream(messages, **stream_options):
                response = chunk if response is None else response + chunk

            if response is None:
                return {"messages": [AIMessage(content=prefix, id=first_id)] if prefix else []}

            usage = response.usage_metadata
            if usage:
                generated_tokens += usage["output_tokens"]

            can_continue = (
                response.response_metadata.get("finish_reason") == "length"
                and not response.tool_call_chunks
                and response.text
                and usage
                and usage["output_tokens"] > 0
            )
            if can_continue and context_window is None:
                # Ask only after a length stop; a restarted server may report a new size.
                context_window = await discover_context_window(model)
            if not (
                can_continue
                and context_window is not None
                and generated_tokens < context_window
                and usage["total_tokens"] < context_window
            ):
                if prefix:
                    response = AIMessage(content=prefix + response.text, id=first_id)
                return {"messages": [response]}

            if not prefix:
                first_id = response.id
            prefix += response.text

            # Use the same chat and an open-ended assistant message to resume mid-answer.
            messages = [system_message, *state["messages"], AIMessage(content=prefix)]
            stream_options["extra_body"] = {
                "continue_final_message": True,
                "add_generation_prompt": False,
            }
            stream_options["tool_choice"] = "none"

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    # Bound repeated tool requests if a model fails to move on to its answer.
    return graph.compile().with_config({"recursion_limit": 12})
