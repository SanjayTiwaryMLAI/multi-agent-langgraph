"""Multi-Agent LangGraph — Research -> Writer -> Critic pipeline"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    topic: str
    research: str
    draft: str
    feedback: str
    final_output: str
    revision_count: int
    decision: str


def researcher_node(state):
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults
    results = TavilySearchResults(max_results=5).invoke(state["topic"])
    context = "\n\n".join([r["content"] for r in results])
    summary = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
        f"Summarize research on: {state['topic']}\n\n{context}"
    ).content
    return {**state, "research": summary}


def writer_node(state):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    fb = f"\nFeedback: {state['feedback']}" if state.get("feedback") else ""
    draft = llm.invoke(
        f"Write an article on: {state['topic']}\nResearch: {state['research']}{fb}"
    ).content
    return {**state, "draft": draft, "revision_count": state.get("revision_count", 0) + 1}


def critic_node(state):
    from langchain_openai import ChatOpenAI
    result = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
        f"Review this article. Respond APPROVE or REVISE: <feedback>\n\n{state['draft']}"
    ).content
    if result.upper().startswith("APPROVE"):
        return {**state, "decision": "approve", "final_output": state["draft"]}
    return {**state, "decision": "revise", "feedback": result.replace("REVISE:", "").strip()}


def should_revise(state):
    return "revise" if state["decision"] == "revise" and state.get("revision_count", 0) < 3 else "finish"


def create_graph():
    wf = StateGraph(AgentState)
    wf.add_node("researcher", researcher_node)
    wf.add_node("writer", writer_node)
    wf.add_node("critic", critic_node)
    wf.set_entry_point("researcher")
    wf.add_edge("researcher", "writer")
    wf.add_edge("writer", "critic")
    wf.add_conditional_edges("critic", should_revise, {"revise": "writer", "finish": END})
    return wf.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = create_graph()
    result = app.invoke(
        {"topic": "LangGraph for AI agents", "revision_count": 0},
        config={"configurable": {"thread_id": "1"}}
    )
    print(result["final_output"])
