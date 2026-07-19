"""Logs the agent as an MLflow model (models-from-code) and registers a new
version in Unity Catalog. Prints the resulting model name/version so a
subsequent step (deploy.py) can pick it up.

Reference: deployment/agent_model.py is the actual servable model definition.
"""
from __future__ import annotations

import os

import mlflow

MODEL_NAME = "cs4603.default.document_analyst_model"  # catalog.schema.model_name


def main() -> None:
    mlflow.set_registry_uri("databricks-uc")
    input_example = {
        "messages": [{"role": "user", "content": "What was FY2023 total revenue?"}]
    }
    with mlflow.start_run(run_name="document_analyst_deploy"):
        logged_model = mlflow.langchain.log_model(
            lc_model="deployment/agent_model.py",
            artifact_path="document_analyst_model",
            input_example=input_example,
        )

    registered = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=MODEL_NAME,
    )

    print(f"Registered model: {MODEL_NAME}")
    print(f"Model version: {registered.version}")

    # Make the new version available to the next workflow step.
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a") as f:
            f.write(f"MODEL_VERSION={registered.version}\n")


if __name__ == "__main__":
    main()