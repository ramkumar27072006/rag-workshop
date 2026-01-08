# GenAI Production Workshop

Hands-on workshop covering GenAI fundamentals and RAG pipelines.

## Notebooks

### 1. GenAI Basics (`01_GenAI_Basics.ipynb`)

| Topic | What You Learn |
|-------|----------------|
| **Tokenization** | How LLMs break text into tokens using `tiktoken` (GPT-4's encoder) |
| **Embeddings** | Convert words to vectors using `sentence-transformers` |
| **Similarity** | Compare word meanings with cosine similarity |
| **Temperature** | Control LLM creativity (0 = factual, 1.5 = creative) |

```python
# Quick Example - Tokenization
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode("Hello AI!")
```

---

### 2. RAG Pipeline (`02_RAG_Pipeline_Advanced.ipynb`)

Build a complete Retrieval-Augmented Generation system:

```
PDF → Chunks → Embeddings → Vector DB → Retrieval → LLM Answer
```

| Step | Tool Used |
|------|-----------|
| Load PDF | `PyPDFLoader` |
| Split Text | `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) |
| Embeddings | `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) |
| Vector Store | `ChromaDB` |
| LLM | `Groq` (llama-3.3-70b-versatile) |

---

## Setup

```bash
pip install -r requirements.txt
```

Set your API key:
```python
import os
os.environ["GROQ_API_KEY"] = "your-key-here"
```

## Requirements

- Python 3.10+
- Groq API Key (free at [groq.com](https://groq.com))
- PDF file for RAG demo
