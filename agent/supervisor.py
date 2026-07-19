"""Supervisor node + routing edge (Task 1.3)."""
from __future__ import annotations

import json
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"

SUPERVISOR_PROMPT = f"""You are an orchestrator coordinating a multi-agent team.
Your task is to classify an atomic task execution step into one of two categories:

1. "{RAG}": Use this when the step requires looking up facts, figures, text, table extraction, or reading specific information from the document.
2. "{MCP}": Use this when the step requires calculations, arithmetic, comparison of parsed data, or numerical data analysis.

Your output must be a single JSON object containing a "next_agent" key with the exact value "{RAG}" or "{MCP}". Do not output anything else.

Example Output:
{{"next_agent": "{RAG}"}}"""


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        current_idx = state.get("current_step_index", 0)

        # 1. Check if all planned steps are complete
        if current_idx >= len(plan):
            return {"next_agent": SYNTH}

        # 2. Extract current step text
        current_step = plan[current_idx]

        # 3. Classify the step using the LLM
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"Classify this step: '{current_step}'")
        ]

        try:
            # Check if LLM supports structured output (with_structured_output)
            # as a fallback, we handle basic JSON parsing
            response = llm.invoke(messages)
            response_text = response.content.strip()
            
            # Simple clean up of potential markdown blocks
            if response_text.startswith("```"):
                import re
                response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text).strip()
                
            parsed = json.loads(response_text)
            next_agent = parsed.get("next_agent", RAG)
            
            # Sanity check validation
            if next_agent not in [RAG, MCP]:
                next_agent = RAG
                
        except Exception as e:
            # Fallback heuristic using keyword analysis in case of model/parse failure
            print(f"Supervisor fallback triggered ({e}). Parsing step via keywords.")
            step_lower = current_step.lower()
            calculation_keywords = [
                "calculate", "compute", "math", "sum", "average", "total", 
                "growth", "rate", "percentage", "compare", "ratio", "difference"
            ]
            if any(kw in step_lower for kw in calculation_keywords):
                next_agent = MCP
            else:
                next_agent = RAG

        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    """Return state['next_agent'] for the conditional edge mapping."""
    next_agent = state.get("next_agent")
    if not next_agent:
        raise ValueError("next_agent was not set in the state prior to routing.")
    return next_agent