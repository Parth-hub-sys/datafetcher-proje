from typing import TypedDict, Optional
import os
import requests
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ------------------ ENV ------------------
load_dotenv()
WEBSEARCH_API_KEY = os.getenv("WEBSEARCH_API_KEY")

from typing import TypedDict

class InputState(TypedDict):
    user_query: str


# ------------------ STATE ------------------
from typing import Optional

class AgentState(InputState):
    need_search: bool
    search_query: Optional[str]
    search_results: Optional[str]
    final_answer: Optional[str]


# ------------------ LLM ------------------
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# ------------------ TOOL ------------------
def websearch_tool(query: str) -> str:
    url = "https://websearchapi.ai/dashboard/api-playground"
    headers = {
        "x-api-key": WEBSEARCH_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"query": query, "num_results": 3}

    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text[:4000]

# ------------------ NODES ------------------
def llm_decision_node(state: AgentState) -> AgentState:
    prompt = f"""
                You are an AI agent.

                User question:
                {state['user_query']}

                Decide:
                1. Is live web search required? (yes or no)
                2. If yes, generate a concise search query.

                Respond strictly in this format:
                need_search: yes/no
                query: <search query or empty>
"""

    response = llm.invoke(prompt).content.lower()

    if "need_search: yes" in response:
        state["need_search"] = True
        state["search_query"] = response.split("query:")[1].strip()
    else:
        state["need_search"] = False
        state["search_query"] = None

    return state

def route_after_decision(state: AgentState):
    if state["need_search"]:
        return "websearch_tool"
    else:
        return "llm_final"




def tool_node(state: AgentState) -> AgentState:
    state["search_results"] = websearch_tool(state["search_query"])
    return state

def compress_results_node(state: AgentState) -> AgentState:
    prompt = f"""
Summarize the following web results into bullet points.
Limit to 200 words.

Results:
{state['search_results']}
"""
    state["search_results"] = llm.invoke(prompt).content
    return state


def llm_final_node(state: AgentState) -> AgentState:
    context = state.get("search_results")

    if not context:
        context = "No external web search was required."

    prompt = f"""
Answer the user's question clearly.

Question:
{state['user_query']}

Context:
{context}
"""

    state["final_answer"] = llm.invoke(prompt).content
    return state



# ------------------ GRAPH ------------------
graph = StateGraph(AgentState,input=InputState)

graph.add_node("llm_decision", llm_decision_node)
graph.add_node("websearch_tool", tool_node)
graph.add_node("compress_results", compress_results_node)
graph.add_node("llm_final", llm_final_node)

graph.set_entry_point("llm_decision")

#  CONDITIONAL EDGE
graph.add_conditional_edges(
    "llm_decision",
    route_after_decision,
    {
        "websearch_tool": "websearch_tool",
        "llm_final": "llm_final"
    }
)

graph.add_edge("websearch_tool", "compress_results")
graph.add_edge("compress_results", "llm_final")
graph.add_edge("llm_final", END)

app = graph.compile()
