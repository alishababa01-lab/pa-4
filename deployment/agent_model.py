"""Self-contained MLflow model definition for the Document Analyst graph.

Reference: databricks_deployment_v1/agent.py

MLflow serves this file via "models-from-code" — it imports this module,
executes it top to bottom, and expects `mlflow.models.set_model(...)` to have
been called with the object to serve. That means every dependency this file
needs must be resolvable from this single file plus ordinary local imports
(agent.graph, rag.store, etc.) — no notebook-only setup, no globals injected
from outside.

Required environment variables (validated below, at import time, before any
client is constructed):
  DATABRICKS_HOST   - workspace URL, e.g. https://<workspace>.databricks.com
  DATABRICKS_TOKEN  - PAT or serving endpoint's own auth token
  DATABRICKS_MODEL  - name of the model serving endpoint to call for chat completions

Must import cleanly:  python -c "import deployment.agent_model"
"""
from __future__ import annotations

import os

import mlflow
from langchain_openai import ChatOpenAI

from agent.graph import build_graph, load_mcp_tools
from rag.store import get_retriever

REQUIRED_ENV_VARS = ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_MODEL"]


def _validate_env() -> None:
    """Fail loudly and specifically at import time.

    Without this, a missing env var surfaces later as a generic
    DEPLOYMENT_FAILED in the serving logs with no indication of *why* — the
    same class of opaque failure we kept hitting with the MCP subprocess
    path issues. Better to raise here, immediately, naming exactly what's
    missing.
    """
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "deployment/agent_model.py: missing required environment "
            f"variable(s): {', '.join(missing)}. Set these on the serving "
            "endpoint (or in your local shell/.env for a dry-run import) "
            "before this model can be built."
        )


def _get_chat_llm() -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at a Databricks model-serving
    endpoint's OpenAI-compatible API.

    Databricks serving endpoints expose an OpenAI-compatible chat
    completions API at {DATABRICKS_HOST}/serving-endpoints — so rather than
    depending on databricks-langchain inside the served model artifact, we
    just point the standard ChatOpenAI client at that URL with the
    Databricks token as the bearer credential.
    """
    databricks_host = os.environ["DATABRICKS_HOST"].rstrip("/")
    return ChatOpenAI(
        model=os.environ["DATABRICKS_MODEL"],
        base_url=f"{databricks_host}/serving-endpoints",
        api_key=os.environ["DATABRICKS_TOKEN"],
        temperature=0,
    )


def _get_mcp_server_path() -> str:
    """tools/mcp_server.py is a sibling of deployment/ and agent/ at the
    repo root — go up one level from this file's directory to reach it."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "tools", "mcp_server.py")


# --- Validate configuration before building anything ------------------------
_validate_env()

# --- Production clients -------------------------------------------------
llm = _get_chat_llm()
retriever = get_retriever(k=4)
tools = load_mcp_tools(server_path=_get_mcp_server_path())

# --- Build and register the graph for MLflow to serve -----------------------
graph = build_graph(llm=llm, retriever=retriever, tools=tools)

mlflow.models.set_model(graph)
