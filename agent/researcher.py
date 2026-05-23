"""Research agent with Tavily web search"""
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


def run_researcher(state: dict) -> dict:
    """Search the web and summarize research on the topic."""
    search = TavilySearchResults(max_results=5)
    results = search.invoke(state["topic"])
    context = "\n\n".join([r["content"] for r in results])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    summary = llm.invoke(
        f"Summarize the following research on '{state['topic']}':\n\n{context}"
    ).content
    return {**state, "research": summary}
