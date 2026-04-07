from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, cast
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from openai import BadRequestError

from tools import calculate_budget, search_flights, search_hotels


BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as file:
        return file.read().strip()


load_dotenv()
SYSTEM_PROMPT = load_system_prompt()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


TOOLS_LIST = [search_flights, search_hotels, calculate_budget]

github_token_raw = os.getenv("GITHUB_TOKEN")
openai_key_raw = os.getenv("OPENAI_API_KEY")

if not github_token_raw and not openai_key_raw:
    raise RuntimeError(
        "Missing API key. Please set GITHUB_TOKEN or OPENAI_API_KEY in the environment (or .env)."
    )

if github_token_raw:
    github_token = cast(str, github_token_raw)

    def get_api_key() -> str:
        return github_token

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=get_api_key,
        base_url="https://models.inference.ai.azure.com",
    )
else:
    assert openai_key_raw is not None
    openai_key = cast(str, openai_key_raw)

    def get_api_key() -> str:
        return openai_key

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=get_api_key,
    )

llm_with_tools = llm.bind_tools(TOOLS_LIST)


JAILBREAK_PATTERN = re.compile(
    r"\b(system prompt|override|bypass|bỏ qua|ignore all|disregard all|in toàn bộ|tiết lộ|secret|hidden instructions)\b",
    re.IGNORECASE,
)


def is_jailbreak_attempt(text: str) -> bool:
    return bool(JAILBREAK_PATTERN.search(text))


def refusal_message() -> AIMessage:
    return AIMessage(
        content=(
            "Xin lỗi, mình không thể làm theo yêu cầu này vì nó không an toàn hoặc có dấu hiệu cố gắng vượt qua hướng dẫn hệ thống. "
            "Nếu bạn cần hỗ trợ về du lịch, đặt vé hoặc đặt phòng, mình sẵn sàng giúp. "
            "Nếu đây là yêu cầu khác, mình sẽ chuyển bạn sang nhân viên hỗ trợ nhé."
        )
    )


def agent_node(state: AgentState):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    latest_user_text = ""
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            latest_user_text = getattr(message, "content", "") or ""
            break

    if latest_user_text and is_jailbreak_attempt(latest_user_text):
        print("Phát hiện prompt injection/jailbreak, từ chối an toàn")
        return {"messages": [refusal_message()]}

    try:
        response = llm_with_tools.invoke(messages)
    except BadRequestError as error:
        error_text = str(error).lower()
        if "content_filter" in error_text or "jailbreak" in error_text:
            print("Azure content filter đã chặn prompt, trả lời an toàn")
            return {"messages": [refusal_message()]}
        raise

    # Logging
    if getattr(response, "tool_calls", None):
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "unknown_tool")
            tool_args = tool_call.get("args", {})
            print(f"Gọi tool: {tool_name}({tool_args})")
    else:
        print("Trả lời trực tiếp")

    return {"messages": [response]}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS_LIST))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()


def run_chat():
    graph = build_graph()
    session_state: AgentState = {"messages": [SystemMessage(content=SYSTEM_PROMPT)]}

    print("=" * 60)
    print("TravelBuddy - Trợ lý Du lịch Thông minh")
    print("Gõ 'quit' để thoát")
    print("=" * 60)

    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        if not user_input:
            continue

        print("\nTravelBuddy đang suy nghĩ...")
        session_state["messages"].append(("human", user_input))
        try:
            result = graph.invoke(session_state)
        except BadRequestError:
            result = {"messages": [refusal_message()]}
        session_state = cast(AgentState, result)
        final_message = result["messages"][-1]
        print(f"\nTravelBuddy: {final_message.content}")


if __name__ == "__main__":
    run_chat()
