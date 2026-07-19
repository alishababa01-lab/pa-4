"""Synthesizer node (Task 1.6)."""
from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.state import AnalystState

SYNTHESIS_PROMPT = """You are a senior investment analyst. Your task is to write a cohesive, professional final response answering the user's initial inquiry.

To help you, a team of specialists has completed a step-by-step investigation and gathered these results:

Investigation Log:
{investigation_context}

CRITICAL INSTRUCTIONS:
1. Synthesize the findings into a clear, logically structured, and professional executive summary.
2. Address any partial failures gracefully (e.g., if a step returned "not found in documents", explain what is missing instead of ignoring it or crashing).
3. Ensure your answer is highly readable and directly resolves the original request.
4. Integrate any citations (like [source: filename, p.N]) provided by the RAG step results naturally into your final synthesis.
"""

def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        # 1. Gather planning steps and their execution results
        plan = state.get("plan", [])
        step_results = state.get("step_results", [])
        
        # 2. Build a clear compilation of what each step discovered
        compiled_findings = []
        for i, step in enumerate(plan):
            result = step_results[i] if i < len(step_results) else "No result (step skipped or failed)"
            compiled_findings.append(f"Step {i+1}: {step}\nResult: {result}")
            
        investigation_context = "\n\n".join(compiled_findings)
        
        # Get the original user request to keep the response focused
        messages = state.get("messages", [])
        user_query = ""
        # Traverse backwards to find the last HumanMessage
        for msg in reversed(messages):
            if msg.type == "human":
                user_query = msg.content
                break
        
        if not user_query:
            user_query = "Please synthesize the gathered analytical findings."

        # 3. Call the LLM to generate the cited final response
        sys_msg = SystemMessage(content=SYNTHESIS_PROMPT.format(investigation_context=investigation_context))
        human_msg = HumanMessage(content=f"Original Request: {user_query}\n\nGenerate the final report.")
        
        try:
            response = llm.invoke([sys_msg, human_msg])
            final_text = response.content.strip()
        except Exception as e:
            print(f"Synthesis generation failed ({e}). Falling back to raw compilation.")
            # Fallback output in case of generation failure
            final_text = (
                "An error occurred during final synthesis. Here are the gathered facts:\n\n" 
                + investigation_context
            )

        # 4. Return state updates matching the serving contract:
        #   - Update "final_answer" for internal scratchpad/evaluations.
        #   - Append an AIMessage to the "messages" channel so MLflow Serving outputs it.
        return {
            "final_answer": final_text,
            "messages": [AIMessage(content=final_text)]
        }

    return synthesizer