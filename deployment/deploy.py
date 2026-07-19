import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

def deploy_or_update_serving_endpoint(
    endpoint_name: str,
    model_name: str,
    model_version: str,
    vs_endpoint: str,
    vs_index: str,
    embeddings_endpoint: str = "databricks-gte-large-en"
):
    """
    Creates or updates a Databricks Model Serving endpoint with secure secret references.
    """
    # Initialize Databricks Workspace Client
    print("Initializing Databricks Workspace Client...")
    w = WorkspaceClient()

    # Define our serving configuration containing the environment variables
    # (Secrets are referenced securely via DBUtils/Secrets syntax in double curly braces)
    environment_vars = {
        "DATABRICKS_HOST":  "{{secrets/cs4603-deploy/DATABRICKS_HOST}}",
        "DATABRICKS_TOKEN": "{{secrets/cs4603-deploy/DATABRICKS_TOKEN}}",
        "DATABRICKS_MODEL": "{{secrets/cs4603-deploy/DATABRICKS_MODEL}}",
        # Non-sensitive variables required for the RAG container's retriever to start successfully:
        "VECTOR_SEARCH_ENDPOINT": vs_endpoint,
        "VECTOR_SEARCH_INDEX":    vs_index,
        "EMBEDDINGS_ENDPOINT":    embeddings_endpoint,
    }

    # Construct the configuration input
    config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                name="document_analyst_entity",  # A local name for your served entity within the endpoint
                entity_name=model_name,
                entity_version=model_version,
                workload_size="Small",
                scale_to_zero_enabled=True,
                environment_vars=environment_vars
            )
        ]
    )

    # Check if the serving endpoint already exists
    endpoint_exists = False
    try:
        print(f"Checking if endpoint '{endpoint_name}' exists...")
        existing_endpoint = w.serving_endpoints.get(name=endpoint_name)
        endpoint_exists = True
        print(f"Endpoint '{endpoint_name}' found. An update will be applied.")
    except Exception as e:
        if "does not exist" in str(e) or "not found" in str(e).lower() or "404" in str(e):
            print(f"Endpoint '{endpoint_name}' does not exist. It will be created.")
        else:
            raise e

    if endpoint_exists:
        # Update existing endpoint configuration
        print(f"Updating configuration for endpoint: {endpoint_name}...")
        # update_config updates the configuration of an existing endpoint in-place
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=config.served_entities
        )
    else:
        # Create a new endpoint
        print(f"Creating new serving endpoint: {endpoint_name}...")
        w.serving_endpoints.create(
            name=endpoint_name,
            config=config
        )

    # Wait for the endpoint to reach READY state
    print(f"Waiting for endpoint '{endpoint_name}' to reach READY state...")
    while True:
        status = w.serving_endpoints.get(name=endpoint_name)
        state = status.state
        
        # Check overall endpoint state
        ready_state = state.ready.value if hasattr(state.ready, "value") else str(state.ready)
        config_update_state = state.config_update.value if hasattr(state.config_update, "value") else str(state.config_update)
        
        print(f"Current State: {ready_state} | Configuration Update: {config_update_state}")
        
        if ready_state == "READY" and config_update_state != "UPDATING":
            print(f"\nEndpoint '{endpoint_name}' is fully deployed and READY!")
            break
        elif ready_state == "FAILED":
            raise RuntimeError(f"Endpoint deployment failed. State details: {status}")
            
        time.sleep(15)

    # Print the endpoint details and endpoint URL
    endpoint_url = f"{w.config.host}/serving-endpoints/{endpoint_name}/invocations"
    print("\n" + "="*50)
    print(f"Endpoint Name: {endpoint_name}")
    print(f"Endpoint URL:  {endpoint_url}")
    print("="*50 + "\n")
    return endpoint_url


if __name__ == "__main__":
    # Example execution (Adjust catalog, schema, and index names based on your UC details)
    deploy_or_update_serving_endpoint(
        endpoint_name="document_analyst_27100316",
        model_name="cs4603.default.document_analyst_model",  # catalog.schema.model_name
        model_version="1",
        vs_endpoint="alisha-vs-endpoint",
        vs_index="cs4603.default.alisha_analyst_index"
    )