# 🐧 TurboVec RAG with LlamaIndex + Ollama

```text
        .--.
       |o_o |     Local RAG
       |:_/ |     4-bit vector compression
      //   \ \    TurboVec + LlamaIndex + Ollama
     (|     | )
    /'\_   _/`\
    \___)=(___/
```

A fully local Retrieval-Augmented Generation demo using **TurboVec/TurboQuant** as the compressed vector store, **LlamaIndex** as the RAG orchestration layer, and **Ollama** for local embeddings and LLM inference.

This project indexes a structured FIFA World Cup 2026 knowledge file, retrieves relevant chunks through TurboVec, and answers questions locally without sending documents, embeddings, or prompts to a hosted service.

---

## Why this project exists

RAG systems store embedding vectors for every chunk of text. Standard `float32` embeddings are memory-heavy, especially when document collections grow.

TurboVec uses TurboQuant-style low-bit vector compression to reduce vector memory while keeping retrieval practical for local AI workflows.

For a 768-dimensional embedding:

```text
float32 vector = 768 × 4 bytes       = 3,072 bytes
4-bit vector  = 768 × 4 / 8 bytes   =   384 bytes
compression   = 8x smaller
```

---

## Architecture

```text
┌──────────────────────────────┐
│ fifa_world_cup_2026_rag.txt  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ LlamaIndex chunking           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Ollama embeddings             │
│ nomic-embed-text              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ TurboVec IdMapIndex           │
│ 4-bit TurboQuant compression  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ LlamaIndex query engine       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Ollama LLM                    │
│ gemma3:4b                     │
└──────────────────────────────┘
```

---

## Repository structure

```text
.
├── rag_turbovec.py
├── compression_stats.py
├── fifa_world_cup_2026_rag_input.txt
└── README.md
```

| File | Purpose |
|---|---|
| `rag_turbovec.py` | Main RAG pipeline using LlamaIndex, Ollama, and TurboVec |
| `compression_stats.py` | Standalone script to estimate float32 vs TurboQuant vector memory |
| `fifa_world_cup_2026_rag_input.txt` | Source document used for retrieval |
| `README.md` | Project documentation |

---

## Prerequisites

Install these before running the project:

- Python 3.10 or newer
- Ollama
- A local embedding model, such as `nomic-embed-text`
- A local LLM, such as `gemma3:4b`

---

## Step 1: Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

For Conda users:

```bash
conda create -n turbovec-rag python=3.11 -y
conda activate turbovec-rag
python -m pip install -U pip
```

---

## Step 2: Install dependencies

```bash
python -m pip install -U "turbovec[llama-index]" \
  llama-index \
  llama-index-llms-ollama \
  llama-index-embeddings-ollama \
  llama-index-readers-file
```

---

## Step 3: Pull Ollama models

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

Make sure Ollama is running:

```bash
ollama list
```

---

## Step 4: Check the input file path

Open `rag_turbovec.py` and confirm this path matches your machine:

```python
WORLD_CUP_FILE = Path("~turbovec/fifa_world_cup_2026_rag_input.txt")
```

If your file is in the same repo folder, you can change it to:

```python
WORLD_CUP_FILE = Path("fifa_world_cup_2026_rag_input.txt")
```

---

## Step 5: Run the RAG pipeline

```bash
python rag_turbovec.py
```

The script asks example questions such as:

```text
How many teams are playing in the FIFA World Cup 2026?

How many groups and total matches are there?

Which teams is Argentina facing in the group stage?

How can Argentina win the World Cup?
```

Expected flow:

```text
Loading document...
Creating TurboVec IdMapIndex...
Indexing document with TurboVec/TurboQuant...
Generating embeddings...
Running RAG questions...
```

---

## Step 6: Run compression statistics

```bash
python compression_stats.py
```

Example output:

```text
--- TurboQuant Compression Statistics ---
Vectors indexed     : 5
Dimensions          : 768
Bit width           : 4-bit
Original size       : 15,360 bytes (15.0 KB) at float32
Compressed size     : 1,920 bytes (1.9 KB) at 4-bit
Compression ratio   : 8.0x smaller
------------------------------------------------
```

The stats estimate raw vector storage only. Actual memory use may include document text, Python objects, metadata, and index overhead.

---

## Key implementation detail

For LlamaIndex integration, use `IdMapIndex`, not raw `TurboQuantIndex`.

```python
from turbovec import IdMapIndex
from turbovec.llama_index import TurboQuantVectorStore

tv_index = IdMapIndex(
    dim=768,
    bit_width=4,
)

vector_store = TurboQuantVectorStore(
    index=tv_index,
    bit_width=4,
    stores_text=True,
    is_embedding_query=True,
)
```

`IdMapIndex` supports stable external IDs through `add_with_ids()`, which is what the LlamaIndex adapter expects.

---

## Troubleshooting

### `No module named llama_index`

Install the LlamaIndex packages inside the same environment used to run the script:

```bash
python -m pip install -U llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-readers-file
```

### Ollama connection error

Confirm Ollama is running:

```bash
ollama list
```

Then pull the required models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

### `add_with_ids` error

Use `IdMapIndex`, not `TurboQuantIndex`, for the LlamaIndex vector store.

Correct:

```python
from turbovec import IdMapIndex
```

Avoid this for the LlamaIndex adapter:

```python
from turbovec import TurboQuantIndex
```

---

## Tech stack

```text
🐧 Linux / macOS shell
🧠 Ollama
🧩 LlamaIndex
⚡ TurboVec / TurboQuant
📄 Local TXT knowledge base
```

---

## Tagline

**Local RAG with TurboVec 4-bit vector compression, LlamaIndex, and Ollama.**
