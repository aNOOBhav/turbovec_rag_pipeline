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

import rag_turbovec


def fmt_size(num_bytes):
    if num_bytes is None:
        return "NULL"
    return f"{num_bytes:,.0f} bytes ({num_bytes / 1024:.1f} KB)"


def print_compression_stats(num_vectors, dim, bit_width):
    print("\n--- TurboQuant Compression Statistics ---")

    if not num_vectors or num_vectors <= 0:
        original_bytes = None
        compressed_bytes = None
        compression_ratio = None
    else:
        original_bytes = num_vectors * dim * 4
        compressed_bytes = num_vectors * dim * bit_width / 8

        if compressed_bytes == 0:
            compression_ratio = None
        else:
            compression_ratio = original_bytes / compressed_bytes

    print(f"Vectors indexed     : {num_vectors if num_vectors else 'NULL'}")
    print(f"Dimensions          : {dim if dim else 'NULL'}")
    print(f"Bit width           : {str(bit_width) + '-bit' if bit_width else 'NULL'}")
    print(f"Original size       : {fmt_size(original_bytes)} at float32")
    print(f"Compressed size     : {fmt_size(compressed_bytes)} at {bit_width}-bit")

    if compression_ratio is None:
        print("Compression ratio   : NULL")
    else:
        print(f"Compression ratio   : {compression_ratio:.1f}x smaller")

    print("-" * 48)


def get_num_vectors(index, documents):
    """
    Try multiple ways to count indexed chunks.
    Some vector stores do not expose nodes_dict cleanly.
    """

    # Most common LlamaIndex count
    try:
        count = len(index.index_struct.nodes_dict)
        if count > 0:
            return count
    except Exception:
        pass

    # Fallback: index struct may expose nodes directly
    try:
        count = len(index.index_struct.nodes)
        if count > 0:
            return count
    except Exception:
        pass

    # Fallback: estimate from docstore
    try:
        count = len(index.docstore.docs)
        if count > 0:
            return count
    except Exception:
        pass

    # Last fallback: at least document count
    try:
        return len(documents)
    except Exception:
        return 0


def build_index_for_stats():
    world_cup_file = rag_turbovec.WORLD_CUP_FILE
    llm_model = rag_turbovec.LLM_MODEL
    embed_model = rag_turbovec.EMBED_MODEL
    embed_dim = rag_turbovec.EMBED_DIM
    bit_width = rag_turbovec.BIT_WIDTH

    if not world_cup_file.exists():
        raise FileNotFoundError(f"Input file not found: {world_cup_file}")

    Settings.llm = Ollama(
        model=llm_model,
        request_timeout=180.0,
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=embed_model,
    )

    Settings.chunk_size = 700
    Settings.chunk_overlap = 100

    print(f"Loading document: {world_cup_file}")
    documents = SimpleDirectoryReader(
        input_files=[str(world_cup_file)]
    ).load_data()

    print("Creating TurboQuant vector index for stats...")
    tv_index = IdMapIndex(
        dim=embed_dim,
        bit_width=bit_width,
    )

    vector_store = TurboQuantVectorStore(
        index=tv_index,
        bit_width=bit_width,
        stores_text=True,
        is_embedding_query=True,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    num_vectors = get_num_vectors(index, documents)

    return num_vectors, embed_dim, bit_width


def main():
    num_vectors, embed_dim, bit_width = build_index_for_stats()

    print_compression_stats(
        num_vectors=num_vectors,
        dim=embed_dim,
        bit_width=bit_width,
    )


if __name__ == "__main__":
    main()