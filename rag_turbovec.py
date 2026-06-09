from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from turbovec import IdMapIndex
from turbovec.llama_index import TurboQuantVectorStore


WORLD_CUP_FILE = Path("/Users/aamajumd/turbovec/fifa_world_cup_2026_rag_input.txt")

LLM_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"

EMBED_DIM = 768
BIT_WIDTH = 4

QUESTIONS = [
    "How many teams are playing in the FIFA World Cup 2026?",
    "How many groups and total matches are there? Split the matches by group stage and knockout rounds.",
    "Which teams is Argentina facing in the group stage?",
    "How can Argentina win the World Cup? Which teams might Argentina need to face and beat? Explain that the path is conditional.",
]

SYSTEM_PROMPT = """
You are answering questions using only the FIFA World Cup 2026 RAG notes.

Rules:
- Be factual and concise.
- Do not invent teams, venues, fixtures, or results.
- If the document does not contain enough information, say that clearly.
- For Argentina's path, explain that future knockout opponents are conditional
  and depend on group standings and bracket progression.
"""


def build_query_engine():
    if not WORLD_CUP_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {WORLD_CUP_FILE}")

    Settings.llm = Ollama(
        model=LLM_MODEL,
        request_timeout=180.0,
        system_prompt=SYSTEM_PROMPT,
    )

    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

    Settings.chunk_size = 700
    Settings.chunk_overlap = 100

    print(f"Loading document: {WORLD_CUP_FILE}")
    documents = SimpleDirectoryReader(
        input_files=[str(WORLD_CUP_FILE)]
    ).load_data()

    print(f"Creating TurboVec IdMapIndex: dim={EMBED_DIM}, bit_width={BIT_WIDTH}")

    tv_index = IdMapIndex(
        dim=EMBED_DIM,
        bit_width=BIT_WIDTH,
    )

    vector_store = TurboQuantVectorStore(
        index=tv_index,
        bit_width=BIT_WIDTH,
        stores_text=True,
        is_embedding_query=True,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    print("Indexing document with TurboVec/TurboQuant...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    return index.as_query_engine(similarity_top_k=5)


def main():
    query_engine = build_query_engine()

    print("\n--- RAG: TurboVec + LlamaIndex + Ollama ---\n")

    for question in QUESTIONS:
        print(f"Q: {question}")
        response = query_engine.query(question)
        print(f"A: {response}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()