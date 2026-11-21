"""Small, provider-independent LangGraph chat graph."""

# LangChain supplies the provider-neutral model and messages; LangGraph defines the workflow.
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import Settings
from app.skills import build_system_prompt, discover_skills, make_load_skill
from app.tools import make_web_search

SKILLS = discover_skills()
BASE_SYSTEM_PROMPT = (
    build_system_prompt(SKILLS)
    + """

When attached sources are provided, cite factual claims with the exact marker
[[cite:SOURCE_ID:PAGE_NUMBER]], using only source IDs and page numbers shown in the
attachment context. Never invent citations. For image attachments, cite page 1. Do not
emit citation markers when no attached source supports the claim."""
)

WEB_SEARCH_PROMPT = """

You have a web_search tool. For any question about the latest, current, or recent
information (versions, releases, news, prices, dates), you MUST call web_search before
answering; do not answer from memory. Do not add years to your queries, because your
memory of the current date may be out of date; use the dates in the results instead.
Search once. If the results look old, lack dates, or do not clearly answer the
question, search again with a different query (for example add "site:" with the
official website), up to three searches in total, then write your answer. When
sources disagree, prefer the newest dated official source. Cite each web-derived
sentence with the exact marker [[web:ID]] at the end of the sentence, using only ids
returned by the tool. Never invent ids or URLs. Do not write a "Sources" section or
raw URLs: the app renders the links from your markers. Example of a correctly cited sentence: Python 3.15.2 is the latest release. [[web:ab12cd34]]"""



def create_graph(model: BaseChatModel, settings: Settings | None = None):
    """Let the model load relevant skills and search the web, then stream its answer."""
    tools = [make_load_skill(SKILLS)]
    system_prompt = BASE_SYSTEM_PROMPT
    if settings is not None and settings.searxng_url:
        tools.append(
            make_web_search(
                settings.searxng_url,
                timeout = settings.web_tool_timeout_s,
                max_results=settings.web_search_max_results,
                max_calls=settings.web_max_calls_per_answer
            )
        )
        system_prompt += WEB_SEARCH_PROMPT
    
    # LCEL formats the prompt, then passes its messages to the configured model.
    
    chat_chain = ChatPromptTemplate.from_messages(
        [
            # Skills can contain JSON/code braces; they are literal text, not templates.
            SystemMessage(content=system_prompt),
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
