"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search."""
from __future__ import annotations

# Standard library imports
import os

# Third-party imports
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AnalystState


def format_docs(docs) -> str:
    """Format retrieved documents with clean source and page citations."""
    if not docs:
        return ""
    
    formatted_chunks = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}
        
        # Parse the source filename from the full DBFS/Volume path
        source_path = metadata.get("source", "unknown")
        filename = os.path.basename(source_path) if "/" in source_path else source_path
        
        # Get the page number (we mapped chunk_position to 'page' during ingestion)
        page = metadata.get("page", "N/A")
        
        chunk_text = doc.page_content.strip()
        formatted_chunks.append(
            f"--- Document Chunk {i+1} [source: {filename}, p.{page}] ---\n{chunk_text}"
        )
        
    return "\n\n".join(formatted_chunks)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        current_idx = state.get("current_step_index", 0)
        
        # Safeguard: If current index is out of bounds, exit early
        if current_idx >= len(plan):
            return {}
            
        current_step = plan[current_idx]
        
        # 1. Retrieve the top-k relevant chunks from Databricks Vector Search
        try:
            retrieved_docs = retriever.invoke(current_step)
        except Exception as e:
            print(f"Vector search retrieval failed ({e})")
            retrieved_docs = []
            
        # 2. Handle empty retrieval gracefully
        if not retrieved_docs:
            fact = "not found in documents"
        else:
            # Format retrieved chunks with metadata citations
            context_str = format_docs(retrieved_docs)
            
            # 3. Use the LLM to extract the precise fact based ONLY on the context
            rag_system_prompt = """You are a highly precise financial and document analyst. 
Your objective is to answer the user's specific step query using only the provided document chunks.

CRITICAL INSTRUCTIONS:
1. Base your answer strictly on the facts present within the provided document chunks.
2. If the chunks do not contain the answer, reply with exactly: "not found in documents"
3. Cite your sources directly in your response using the format [source: filename, p.N] as shown in the document chunk headers.

Document Chunks:
{context}"""

            messages = [
                SystemMessage(content=rag_system_prompt.format(context=context_str)),
                HumanMessage(content=f"Answer this specific step query: {current_step}")
            ]
            
            try:
                response = llm.invoke(messages)
                fact = response.content.strip()
            except Exception as e:
                print(f"LLM fact extraction failed ({e})")
                fact = "not found in documents"

        # 4. Prepare updates: append result and increment the step index
        updated_results = list(state.get("step_results", [])) + [fact]
        
        return {
            "step_results": updated_results,
            "current_step_index": current_idx + 1
        }

    return rag_agent