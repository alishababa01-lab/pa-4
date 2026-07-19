"""All system prompts for the Document Analyst (single source of truth).

TODO: Write clear system prompts for each node. Keep them here so behaviour is
tunable without touching node logic.
"""

PLANNER_PROMPT = ""  # TODO: decompose the query into a JSON array of 2-5 steps
SUPERVISOR_PROMPT = ""  # TODO: classify a step -> 'rag_agent' or 'mcp_tools'
RAG_EXTRACT_PROMPT = ""  # TODO: extract one cited fact from retrieved chunks
MCP_STEP_PROMPT = ""  # TODO: instruct the model to call exactly one math tool
SYNTHESIZER_PROMPT = ""  # TODO: combine step results into a cited final answer
