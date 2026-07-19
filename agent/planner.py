from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AnalystState

# Prompt instructing the LLM to structure output as a clean JSON list of 2-5 atomic steps
PLANNER_PROMPT = """You are an expert planning assistant. Your job is to decompose analytical financial or document queries into 2 to 5 ordered atomic steps.

Decompose the request into:
1. Retrieval steps (extracting specific facts, figures, or tables from the documents).
2. Calculation/Synthesis steps (comparing numbers, calculating growth rates, or compiling final trends).

CRITICAL: Your output must be a valid JSON array of strings, containing only the steps. Do not include any markdown formatting, code blocks (like ```json), explanation, or other text outside the JSON array.

Example Output:
["Retrieve the total revenue for FY2022 and FY2023 from the annual report.", "Compute the year-over-year revenue growth rate."]"""


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        # 1. Retrieve the latest user query from the messages list
        # Typically the last message, or the first HumanMessage
        user_query = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                user_query = msg.content
                break
        
        # Fallback if no message content is found
        if not user_query:
            user_query = "Please analyze the loaded documents."

        # 2. Invoke the LLM
        messages = [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=f"Decompose this query: {user_query}")
        ]
        
        try:
            response = llm.invoke(messages)
            response_text = response.content.strip()
            
            # Clean up potential markdown formatting block wrapper (```json ... ```)
            if response_text.startswith("```"):
                # Remove code blocks if present
                response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text).strip()

            # 3. Parse JSON safely
            plan = json.loads(response_text)
            
            if not isinstance(plan, list):
                raise ValueError("Output is not a JSON list")
                
            # Ensure steps are strings and size constraints are respected
            plan = [str(step).strip() for step in plan if step]
            if not plan:
                raise ValueError("Plan list is empty")
                
        except Exception as e:
            # Fallback gracefully to a single-step execution plan on parse failure
            print(f"Planner parsing failed ({e}). Falling back to a single-step plan.")
            plan = [f"Directly answer the query: {user_query}"]

        # 4. Return updates matching the schema requirements
        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": []
        }

    return planner
