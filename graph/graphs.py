# from typing import TypedDict
# import os
# import requests
# from dotenv import load_dotenv

# from langgraph.graph import StateGraph, END


# # -------------------------------------------------
# # Load environment variables (VERY IMPORTANT)
# # -------------------------------------------------
# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# WEBSEARCH_API_KEY = os.getenv("WEBSEARCH_API_KEY")

# if not GROQ_API_KEY:
#     raise EnvironmentError("OPENAI_API_KEY not set")

# if not WEBSEARCH_API_KEY:
#     raise EnvironmentError("WEBSEARCH_API_KEY not set")

# # -------------------------------------------------
# # Public input state (LangSmith UI)
# # -------------------------------------------------
# class InputState(TypedDict):
#     user_query: str

# # -------------------------------------------------
# # Internal / output state
# # -------------------------------------------------
# class AgentState(InputState):
#     final_answer: str

# # -------------------------------------------------
# # LLM
# # -------------------------------------------------
# from langchain_groq import ChatGroq

# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0
# )
# # -------------------------------------------------
# # WebSearchAPI tool
# # -------------------------------------------------
# def websearch_tool(query: str) -> str:
#     url = "https://websearchapi.ai/dashboard/api-playground"
#     headers = {
#         "x-api-key": WEBSEARCH_API_KEY,
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "query": query,
#         "num_results": 2
#     }

#     response = requests.post(url, json=payload, headers=headers, timeout=30)
#     response.raise_for_status()

#     # HARD LIMIT to avoid token explosion
#     return response.text[:4000]

# # -------------------------------------------------
# # SINGLE NODE: llm_decision (does EVERYTHING)
# # -------------------------------------------------
# def llm_decision_node(state: AgentState) -> AgentState:
#     user_query = state["user_query"]

#     # 1️⃣ Decide if web search is required
#     decision_prompt = f"""
# You are an AI agent.

# User question:
# {user_query}

# Do you need live web search to answer this?
# Answer with ONLY: yes or no.
# """

#     decision = llm.invoke(decision_prompt).content.strip().lower()

#     # 2️⃣ If NO search → answer directly
#     if decision == "no":
#         answer_prompt = f"""
#                         Answer the following question clearly and concisely:

#                         {user_query}
#                         """
#         state["final_answer"] = llm.invoke(answer_prompt).content
#         return state

#     # 3️⃣ Generate search query
#     search_prompt = f"""
#                         Generate a concise web search query for:

#                         {user_query}
#                         """
#     search_query = llm.invoke(search_prompt).content.strip()

#     # 4️⃣ Call WebSearchAPI
#     raw_results = websearch_tool(search_query)

#     # 5️⃣ Compress results
#     compress_prompt = f"""
# Summarize the following web results into bullet points.
# Limit to 200 words.

# Results:
# {raw_results}
# """
#     compressed_results = llm.invoke(compress_prompt).content

#     # 6️⃣ Final answer
#     final_prompt = f"""
# Answer the user's question using the context below.

# Question:
# {user_query}

# Context:
# {compressed_results}
# """
#     state["final_answer"] = llm.invoke(final_prompt).content
#     return state

# # -------------------------------------------------
# # Build LangGraph (ONE NODE ONLY)
# # -------------------------------------------------
# graph = StateGraph(
#     AgentState,
#     input=InputState   #  ONLY user_query is exposed
# )

# graph.add_node("llm_decision", llm_decision_node)
# graph.set_entry_point("llm_decision")
# graph.add_edge("llm_decision", END)

# # REQUIRED variable name
# app = graph.compile()


# graph/graph.py

from typing import TypedDict, Literal
import os
import requests
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

# -------------------------------------------------
# Env setup
# -------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEBSEARCH_API_KEY = os.getenv("WEBSEARCH_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set")

if not WEBSEARCH_API_KEY:
    raise EnvironmentError("WEBSEARCH_API_KEY not set")

# -------------------------------------------------
# State definitions
# -------------------------------------------------
class InputState(TypedDict):
    user_query: str

class AgentState(InputState):
    decision: Literal["yes", "no"]
    search_query: str
    web_results: str
    final_answer: str

# -------------------------------------------------
# Lazy LLM loader (IMPORTANT)
# -------------------------------------------------
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

# -------------------------------------------------
# TOOL NODE: Web Search
# -------------------------------------------------
def websearch_tool_node(state: AgentState) -> AgentState:
    print("WEB SEARCH NODE")

    try:
        url = "https://api.websearchapi.ai/search"
        headers = {
            "x-api-key": WEBSEARCH_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "q": state["search_query"],
            "num_results": 3
        }

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=8
        )
        response.raise_for_status()

        state["web_results"] = response.text[:3000]

    except Exception as e:
        state["web_results"] = f"Web search failed: {str(e)}"

    return state

# -------------------------------------------------
# NODE 1: Decide whether tool is needed
# -------------------------------------------------
def decision_node(state: AgentState) -> AgentState:
    print("DECISION NODE")

    llm = get_llm()
    user_query = state["user_query"]

    prompt = f"""
You are an AI agent.

User question:
{user_query}

Do you need live web search to answer this?
Reply ONLY with: yes or no.
"""

    decision = llm.invoke(prompt).content.strip().lower()
    state["decision"] = "yes" if decision == "yes" else "no"

    return state

# -------------------------------------------------
# NODE 2: Generate search query (only if needed)
# -------------------------------------------------
def search_query_node(state: AgentState) -> AgentState:
    print("SEARCH QUERY NODE")

    llm = get_llm()

    prompt = f"Generate a concise web search query for: {state['user_query']}"
    state["search_query"] = llm.invoke(prompt).content.strip()

    return state

# -------------------------------------------------
# NODE 3: Final answer
# -------------------------------------------------
def answer_node(state: AgentState) -> AgentState:
    print("ANSWER NODE")

    llm = get_llm()

    # If no web search → direct answer
    if state["decision"] == "no":
        state["final_answer"] = llm.invoke(
            state["user_query"]
        ).content
        return state

    # With web context
    prompt = f"""
Answer the user's question using the context below.

Question:
{state['user_query']}

Context:
{state['web_results']}
"""
    state["final_answer"] = llm.invoke(prompt).content
    return state

# -------------------------------------------------
# ROUTER: decide next step
# -------------------------------------------------
def route_after_decision(state: AgentState):
    return "search_query" if state["decision"] == "yes" else "answer"

# -------------------------------------------------
# Build LangGraph
# -------------------------------------------------
graph = StateGraph(AgentState, input=InputState)

graph.add_node("decision", decision_node)
graph.add_node("search_query", search_query_node)
graph.add_node("websearch", websearch_tool_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("decision")

graph.add_conditional_edges(
    "decision",
    route_after_decision,
    {
        "search_query": "search_query",
        "answer": "answer"
    }
)

graph.add_edge("search_query", "websearch")
graph.add_edge("websearch", "answer")
graph.add_edge("answer", END)

# REQUIRED
app = graph.compile()
