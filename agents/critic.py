"""Critic agent that reviews drafts"""
from langchain_openai import ChatOpenAI


class CriticDecision:
    APPROVE = "approve"
    REVISE  = "revise"


def run_critic(state: dict) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = llm.invoke(
        f"Review this article. Respond APPROVE or REVISE: <feedback>\n\n{state['draft']}"
    ).content
    if result.upper().startswith("APPROVE"):
        return {**state, "decision": CriticDecision.APPROVE, "final_output": state["draft"], "feedback": ""}
    return {**state, "decision": CriticDecision.REVISE, "feedback": result.replace("REVISE:", "").strip()}
