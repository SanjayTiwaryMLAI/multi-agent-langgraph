"""Content writer agent"""
from langchain_openai import ChatOpenAI


def run_writer(state: dict) -> dict:
    """Draft an article based on research, incorporating any feedback."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    feedback_note = f"\nFeedback to address: {state['feedback']}" if state.get("feedback") else ""
    draft = llm.invoke(
        f"Write a detailed article on: '{state['topic']}'\n\n"
        f"Research:\n{state['research']}{feedback_note}\n\n"
        f"Include introduction, key sections, and conclusion."
    ).content
    return {**state, "draft": draft, "revision_count": state.get("revision_count", 0) + 1}
