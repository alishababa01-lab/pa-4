"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).
 
Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirrors PA2 Part 1:
 
  - `build_chunks_table(spark, volume_path, chunks_table)`: parses the PDF with
    ai_parse_document, chunks with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enables Change Data
    Feed on the table.
  - `create_index()`: creates a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
  - `ingest(spark, volume_path)`: end-to-end helper that runs both steps and
    blocks until the index reaches READY.
"""
 
from __future__ import annotations
 
import os
import time
 
from config import get_settings
 
# ─── Config pulled from the .env / secret scope ──────────────────────────────
_settings = get_settings()
 
UC_CATALOG = os.environ.get("UC_CATALOG", "main")
UC_SCHEMA = os.environ.get("UC_SCHEMA", "default")
SOURCE_TABLE = os.environ.get("SOURCE_TABLE", f"{UC_CATALOG}.{UC_SCHEMA}.annual_report_chunks")
VECTOR_SEARCH_ENDPOINT = _settings["vs_endpoint"] or os.environ["VECTOR_SEARCH_ENDPOINT"]
VECTOR_SEARCH_INDEX = _settings["vs_index"] or os.environ["VECTOR_SEARCH_INDEX"]
EMBEDDINGS_ENDPOINT = _settings["embeddings"]
 
 
def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse a PDF (or folder of PDFs) in a UC volume and chunk it into a Delta table.

    Args:
        spark: active SparkSession (provided by the Databricks notebook).
        volume_path: path to the PDF, e.g. '/Volumes/main/default/pa4/annual_report.pdf'.
        chunks_table: fully-qualified Delta table name to (re)create, e.g.
            'main.default.annual_report_chunks'.
    """
    parse_table = f"{chunks_table}_parsed"

    # ── Step 1: parse raw PDF bytes into structured content ─────────────────
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {parse_table} (
            path   STRING,
            parsed VARIANT
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    # Check path format and load safely
    if volume_path.lower().endswith(".pdf"):
        print(f"Loading single PDF file directly: {volume_path}")
        df = spark.read.format("binaryFile").load(volume_path)
    else:
        base_dir, glob_pattern = volume_path.rstrip("/"), "*.pdf"
        print(f"Scanning directory: {base_dir} with pattern: {glob_pattern}")
        df = (spark.read.format("binaryFile")
              .option("pathGlobFilter", glob_pattern)
              .load(base_dir))

    # Apply the ai_parse_document function
    parsed_df = df.selectExpr("path", "ai_parse_document(content) AS parsed")

    # Overwrite the parsed table
    parsed_df.write.mode("overwrite").saveAsTable(parse_table)

    # Verify step 1 has written records
    parsed_count = spark.table(parse_table).count()
    print(f"Documents parsed and written to temporary table: {parsed_count}")
    if parsed_count == 0:
        print("Warning: Step 1 parsed 0 documents. Please check if the source PDF is empty or corrupted.")

    # ── Step 2: chunk the parsed documents for retrieval + embedding ────────
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            chunk_id          STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed    STRING,
            source            STRING,
            page              INT
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    spark.sql(f"""
    INSERT OVERWRITE {chunks_table}
    SELECT
        chunk.chunk_id            AS chunk_id,
        chunk.chunk_to_retrieve   AS chunk_to_retrieve,
        chunk.chunk_to_embed      AS chunk_to_embed,
        path                      AS source,
        chunk.chunk_position      AS page  -- mapping chunk_position to the page placeholder
    FROM {parse_table}
    LATERAL VIEW explode(
        from_json(
            cast(ai_prep_search(parsed) as STRING),
            'STRUCT<document: STRUCT<contents: ARRAY<STRUCT<chunk_id: STRING, chunk_to_retrieve: STRING, chunk_to_embed: STRING, chunk_position: INT>>>>'
        ).document.contents
    ) AS chunk
""")

    print(f"Chunks table ready: {chunks_table}")
    print(spark.sql(f"SELECT count(*) AS n_chunks FROM {chunks_table}").toPandas())
 
 
def create_index() -> None:
    """Create (or reuse) a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index with managed embeddings on top of SOURCE_TABLE.
    """
    from databricks.vector_search.client import VectorSearchClient
 
    vsc = VectorSearchClient()
 
    # ── Endpoint ─────────────────────────────────────────────────────────────
    existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if VECTOR_SEARCH_ENDPOINT not in existing_endpoints:
        print(f"Creating endpoint '{VECTOR_SEARCH_ENDPOINT}' ...")
        vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT, endpoint_type="STANDARD")
    else:
        print(f"Endpoint '{VECTOR_SEARCH_ENDPOINT}' already exists.")
 
    # Wait for the endpoint to come online before creating an index on it.
    while True:
        status = vsc.get_endpoint(VECTOR_SEARCH_ENDPOINT).get("endpoint_status", {}).get("state")
        print(f"Endpoint status: {status}")
        if status == "ONLINE":
            break
        time.sleep(10)
 
    # ── Delta Sync index with managed (Databricks-hosted) embeddings ────────
    existing_indexes = [
        i["name"] for i in vsc.list_indexes(VECTOR_SEARCH_ENDPOINT).get("vector_indexes", [])
    ]
    if VECTOR_SEARCH_INDEX in existing_indexes:
        print(f"Index '{VECTOR_SEARCH_INDEX}' already exists — syncing.")
        vsc.get_index(VECTOR_SEARCH_ENDPOINT, VECTOR_SEARCH_INDEX).sync()
        return
 
    print(f"Creating index '{VECTOR_SEARCH_INDEX}' on '{SOURCE_TABLE}' ...")
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_SEARCH_INDEX,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_to_retrieve",
        embedding_model_endpoint_name=EMBEDDINGS_ENDPOINT,
    )
 
    _wait_until_ready(vsc)
 
 
def _wait_until_ready(vsc, timeout_s: int = 900, poll_s: int = 15) -> None:
    """Poll the index until it reaches READY (or raise after timeout_s seconds)."""
    start = time.time()
    while time.time() - start < timeout_s:
        idx = vsc.get_index(VECTOR_SEARCH_ENDPOINT, VECTOR_SEARCH_INDEX)
        status = idx.describe().get("status", {})
        state = status.get("detailed_state") or status.get("state")
        print(f"Index state: {state}")
        if state and "READY" in state:
            print("Index is READY.")
            return
        if state and "FAILED" in state:
            raise RuntimeError(f"Index creation failed: {status}")
        time.sleep(poll_s)
    raise TimeoutError(f"Index did not reach READY within {timeout_s}s")
 
 
def similarity_search_smoke_test(query: str = "What was total revenue in FY2023?", k: int = 3):
    """Quick sanity check: run a similarity search against the live index."""
    from databricks.vector_search.client import VectorSearchClient
 
    vsc = VectorSearchClient()
    idx = vsc.get_index(VECTOR_SEARCH_ENDPOINT, VECTOR_SEARCH_INDEX)
    results = idx.similarity_search(
        query_text=query,
        columns=["chunk_id", "chunk_to_retrieve", "source", "page"],
        num_results=k,
    )
    print(results)
    return results
 
 
def ingest(spark, volume_path: str) -> None:
    """End-to-end: parse + chunk the PDF into SOURCE_TABLE, then build/refresh
    the Vector Search index and block until it's READY.
    """
    build_chunks_table(spark, volume_path, SOURCE_TABLE)
    create_index()
    similarity_search_smoke_test()
 
 
if __name__ == "__main__":
    # Convenience entry point when running this file directly as a Databricks
    # notebook cell / %run target: `spark` is provided by the notebook runtime.
    ingest(spark, volume_path=f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/pa4/annual_report.pdf")