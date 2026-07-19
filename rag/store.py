from __future__ import annotations

from config import get_settings
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import DatabricksVectorSearch

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


def get_vector_store() -> DatabricksVectorSearch:
    """Initialize and return a DatabricksVectorSearch handle (Task 1.4)."""
    settings = get_settings()
    
    endpoint_name = settings["vs_endpoint"]
    index_name = settings["vs_index"]

    # REMOVE text_column=TEXT_COLUMN as the index already defines it as 'chunk_to_retrieve'
    return DatabricksVectorSearch(
        endpoint=endpoint_name,
        index_name=index_name,
        columns=CITATION_COLUMNS
    )


def get_retriever(k: int = 4):
    """Initialize and return a top-k retriever over the index (Task 1.4)."""
    # 1. Obtain the vector store connection
    vector_store = get_vector_store()
    
    # 2. Return it as a retriever configured with k search limits
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )
