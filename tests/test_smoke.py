"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

TODO (Task 1.7 / Bonus A):
  - Build fake LLM / retriever / tool objects (no Databricks, no network).
  - Call `build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])`.
  - Invoke it on a combined retrieval+calculation query and assert that a plan was
    produced, both specialists ran, and the final answer surfaced on messages[-1].

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401 these are instructions for run_testsmoe.y


# ---------------------------------------------------------------------------
# Fakes — no Databricks, no network, no subprocess.
# ---------------------------------------------------------------------------

class FakeDoc:
    """Minimal stand-in for a langchain Document returned by the retriever."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FakeRetriever:
    """Stand-in for the Databricks vector-search retriever (Task 1.4)."""

    def invoke(self, query: str):
        return [
            FakeDoc(
                page_content="Total revenue for fiscal year 2023 was $2.0 billion.",
                metadata={"source": "annual_report_2023.pdf", "page": 8},
            )
        ]


class FakeTool:
    """Stand-in for one MCP tool exposed by the stdio tool server (Task 1.5)."""

    name = "apply_percentage_increase"
    description = "Apply a percentage increase to a base numeric value."

    def invoke(self, args: dict) -> str:
        base = float(args.get("base_value", 0))
        pct = float(args.get("percent", 0))
        return f"{base * (1 + pct / 100):,.0f}"


class FakeLLM:
    """A single fake chat model reused by every node (planner, supervisor,
    rag_agent, mcp_tools, synthesizer). It inspects the system prompt text to
    decide which node is calling it and returns a canned, well-formed
    response for a *combined* retrieval + calculation query — no network
    calls, no API key needed.
    """

    def bind_tools(self, tools):
        # mcp_tools calls llm.bind_tools(tools); keep the same fake instance
        # so invoke() below can still branch on prompt content.
        self._tools = tools
        return self

    def invoke(self, messages):
        text = " ".join(getattr(m, "content", "") or "" for m in messages).lower()

        if "expert planning assistant" in text:
            # Task 1.2: two steps — one retrieval, one calculation
            return _AIMessage(
                content=(
                    '["Retrieve the total revenue for fiscal year 2023 from the document.", '
                    '"Compute what a 10% increase on that revenue would look like."]'
                )
            )

        if "orchestrator coordinating" in text:
            # Task 1.3: route step 1 to RAG, step 2 to MCP.
            if "revenue for fiscal year 2023" in text:
                return _AIMessage(content='{"next_agent": "rag_agent"}')
            return _AIMessage(content='{"next_agent": "mcp_tools"}')

        if "financial and document analyst" in text:
            # Task 1.4: fact extraction over retrieved chunks
            return _AIMessage(
                content="Revenue for 2023 was $2.0 billion. [source: annual_report_2023.pdf, p.8]"
            )

        if "calculation specialist" in text:
            # Task 1.5: force exactly one tool call
            return _AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "apply_percentage_increase",
                        "args": {"base_value": 2_000_000_000, "percent": 10},
                        "id": "call_1",
                    }
                ],
            )

        if "senior investment analyst" in text:
            # Task 1.6: final synthesis
            return _AIMessage(
                content=(
                    "Revenue for fiscal year 2023 was $2.0 billion "
                    "[source: annual_report_2023.pdf, p.8]. A 10% increase would bring "
                    "that to approximately $2.2 billion."
                )
            )

        return _AIMessage(content="ok")


def _AIMessage(content: str, tool_calls=None):
    """Build an AIMessage lazily so this file only needs langchain_core at
    call time (keeps the import-guard test above independent of it)."""
    from langchain_core.messages import AIMessage

    return AIMessage(content=content, tool_calls=tool_calls or [])


# ---------------------------------------------------------------------------
# The actual smoke test
# ---------------------------------------------------------------------------

def test_combined_query_runs_both_specialists_and_produces_final_answer():
    from langchain_core.messages import AIMessage, HumanMessage

    from agent.graph import build_graph

    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])

    query = "What was the revenue in 2023, and what would a 10% increase look like?"
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "plan": [],
        "current_step_index": 0,
        "step_results": [],
        "next_agent": "",
        "final_answer": "",
    }

    result = graph.invoke(initial_state)

    # A plan was produced with both a retrieval and a calculation step.
    assert len(result["plan"]) == 2, f"Expected a 2-step plan, got: {result['plan']}"

    # Both specialists ran — one step_result per planned step.
    assert len(result["step_results"]) == 2, (
        f"Expected both specialists to contribute a result, got: {result['step_results']}"
    )
    assert "not found in documents" not in result["step_results"][0].lower()
    assert result["step_results"][1] != ""

    # The final answer surfaced on messages[-1], satisfying the MLflow
    # messages-in/messages-out serving contract (Task 1.6).
    assert result["messages"], "Expected a non-empty messages list in the final state"
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content.strip() != ""
    assert result["final_answer"].strip() != ""