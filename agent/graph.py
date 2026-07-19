"""Full Document Analyst graph (Tasks 1.5 + 1.7)."""
from __future__ import annotations

import os
import sys
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


#refresh
def load_mcp_tools(server_path: str | None = None) -> list[Any]:
    """Connect the GIVEN MCP server over stdio and return its tools (Task 1.5)."""
    if not server_path:
        # Default fallback: tools/mcp_server.py lives as a SIBLING of agent/,
        # not inside it — so go up one level from this file's directory
        # (agent/) to the repo root, then down into tools/.
        # NOTE: this default only works when graph.py is imported as a real
        # module (__file__ is defined). Inside a Databricks/Jupyter notebook
        # cell __file__ does not exist at all — always pass server_path
        # explicitly there instead of relying on this default.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        server_path = os.path.join(repo_root, "tools", "mcp_server.py")

    server_path = os.path.abspath(server_path)

    # Fail fast and clearly instead of letting a bad path surface later as an
    # opaque asyncio TaskGroup error from a subprocess that never started.
    if not os.path.exists(server_path):
        print(f"MCP server script not found at: {server_path}")
        return []

    # Configure the stdio MCP client
    mcp_config = {
        "computation_server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [server_path]
        }
    }

    try:
        import asyncio
        client = MultiServerMCPClient(mcp_config)

        # Handle async execution safely across both standard and notebook contexts
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're already inside a running event loop (e.g. some notebook
            # kernels). asyncio.run() would raise here — patch the loop with
            # nest_asyncio so it can be re-entered, then drive the coroutine
            # on that SAME loop instead of trying to start a new one.
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(client.get_tools())
        else:
            return asyncio.run(client.get_tools())

    except Exception as e:
        # Surface the real cause instead of swallowing it silently — the
        # generic "unhandled errors in a TaskGroup" message from asyncio
        # hides what actually broke inside the subprocess/handshake.
        import traceback
        print(f"Failed to load MCP tools from {server_path}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return []


def make_mcp_node(tools, llm):
    """Execute one calculation step using exactly one MCP tool (Task 1.5)."""
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def _invoke_tool(tool_obj, args: dict):
        """MCP tools loaded via langchain_mcp_adapters are async-only —
        calling .invoke() directly raises 'StructuredTool does not support
        sync invocation'. Drive .ainvoke() on an event loop instead, handling
        both plain-script and already-running-loop (notebook) contexts."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(tool_obj.ainvoke(args))
        else:
            return asyncio.run(tool_obj.ainvoke(args))

    def mcp_tools(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        current_idx = state.get("current_step_index", 0)

        if current_idx >= len(plan):
            return {}

        current_step = plan[current_idx]

        # 1. Gather all results retrieved in prior steps (e.g., RAG results)
        step_results = state.get("step_results", [])
        context_str = ""
        if step_results:
            context_str = "\n".join(
                f"Step {i+1} Result: {res}" for i, res in enumerate(step_results)
            )

        # 2. Package the prior context with the current calculation directive
        user_content = f"Calculation Step: {current_step}"
        if context_str:
            user_content = (
                f"Context from previous steps:\n{context_str}\n\n"
                f"Use the context above to extract numbers needed for the following task:\n{user_content}"
            )

        messages = [
            SystemMessage(
                content=(
                    "You are a calculation specialist. Choose and call EXACTLY ONE tool "
                    "from your available tools to solve this specific calculation step. "
                    "Analyze the provided context carefully to find the correct figures to use."
                )
            ),
            HumanMessage(content=user_content)
        ]

        result_str = ""
        try:
            if not tools:
                raise ValueError("No tools registered or loaded in the MCP Node.")

            response = llm_with_tools.invoke(messages)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_obj = next((t for t in tools if t.name == tool_call["name"]), None)

                if tool_obj:
                    print(f"Invoking MCP Tool '{tool_call['name']}' with: {tool_call['args']}")
                    tool_out = _invoke_tool(tool_obj, tool_call["args"])
                    result_str = str(tool_out)
                else:
                    result_str = f"Error: Tool '{tool_call['name']}' was selected but not found."
            else:
                result_str = response.content.strip()

        except Exception as e:
            print(f" MCP node failed ({e}). Executing math step via fallback LLM.")
            try:
                # Fallback context mapping
                fallback_user_content = f"Calculate: {current_step}"
                if context_str:
                    fallback_user_content = f"Context from previous steps:\n{context_str}\n\n{fallback_user_content}"

                fallback_resp = llm.invoke([
                    SystemMessage(content="You are a fallback calculation processor. Work out the math."),
                    HumanMessage(content=fallback_user_content)
                ])
                result_str = fallback_resp.content.strip()
            except Exception as le:
                result_str = f"Calculation failed: {le}"

        updated_results = list(state.get("step_results", [])) + [result_str]

        return {
            "step_results": updated_results,
            "current_step_index": current_idx + 1
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble and compile the multi-agent graph (Task 1.7)."""
    if llm is None:
        raise ValueError("A valid Language Model (llm) must be provided to build the graph.")

    # 1. Initialize StateGraph
    builder = StateGraph(AnalystState)

    # 2. Instantiate nodes with injected dependencies
    planner_node = make_planner(llm)
    supervisor_node = make_supervisor(llm)
    rag_node = make_rag_agent(retriever, llm)
    mcp_node = make_mcp_node(tools, llm)
    synthesizer_node = make_synthesizer(llm)

    # 3. Add nodes to the builder
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node(RAG, rag_node)
    builder.add_node(MCP, mcp_node)
    builder.add_node("synthesizer", synthesizer_node)

    # 4. Add static transitions and pathways
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")

    # After agents process a step, they route back to the supervisor
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")

    # Synthesizer is the terminal node
    builder.add_edge("synthesizer", END)

    # 5. Add conditional routing edges out of the supervisor
    builder.add_conditional_edges(
        source="supervisor",
        path=route_from_supervisor,
        path_map={
            RAG: RAG,
            MCP: MCP,
            SYNTH: "synthesizer"
        }
    )

    # 6. Compile graph
    return builder.compile()
