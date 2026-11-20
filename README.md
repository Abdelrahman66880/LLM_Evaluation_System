# PDF RAG Evaluation System

A Python project for asking questions over a collection of PDFs and measuring answer quality. It combines PDF extraction, semantic retrieval, grounded generation, and LLM-as-a-judge evaluation in one installable command-line application.

## What it does

- Extracts and chunks text from one PDF or a directory of PDFs.
- Creates and persists a local vector index; unchanged PDFs reuse their cached embeddings.
- Retrieves source-attributed passages for a user question.
- Generates an answer using only those passages through OpenRouter.
- Generates predictions from a JSONL dataset and scores them with DeepEval metrics: answer relevancy, faithfulness, contextual relevancy, and correctness.
- Writes prediction and evaluation records as JSONL so each stage can be inspected or rerun separately.

## Architecture

```text
PDF files ──> extraction/chunking ──> OpenRouter embeddings ──> local vector index
                                                               │
Question ──> retrieval ──> grounded prompt ──> OpenRouter chat ─┘
                                      │
                           answer + retrieved context
                                      │
Dataset ──> prediction JSONL ──> DeepEval/OpenRouter judge ──> scored JSONL
```

Application packages now live directly in the repository root: `cli/`, `config/`, `schemas/`, `rag/`, `providers/`, and `evaluation/`. Use the CLI below for all RAG and evaluation workflows.

## Requirements and setup

- Python 3.13 or later
- An OpenRouter API key
- `uv` for environment and dependency management

```bash
uv sync
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. The embedding, answer, and judge models can be configured independently:

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_CHAT_MODEL=openai/gpt-4o-mini
OPENROUTER_EVAL_MODEL=openai/gpt-4o-mini
RAG_PDF_PATH=PDFs
RAG_INDEX_PATH=.data/vector_index.json
```

The project uses OpenRouter for both embeddings and chat. It does not require an OpenAI API key. Model availability, pricing, and embedding dimensions depend on the selected OpenRouter model.

## CLI

Run these commands from the repository root after `uv sync`:

```bash
# Build or reuse the local PDF index
uv run llm-evaluation-system index

# Ask a question
uv run llm-evaluation-system ask "How do penguins stay warm?"

# Generate model answers for the sample dataset
uv run llm-evaluation-system generate \
  --dataset datasets/eval_dataset.jsonl \
  --output data/predictions.jsonl

# Score existing predictions with DeepEval
uv run llm-evaluation-system score \
  --input data/predictions.jsonl \
  --output data/evaluation_results.jsonl

# Run generation and scoring sequentially
uv run llm-evaluation-system evaluate \
  --dataset datasets/eval_dataset.jsonl \
  --predictions data/predictions.jsonl \
  --output data/evaluation_results.jsonl
```

Use `--limit 2` with `generate`, `score`, or `evaluate` to run a small sample before processing the full dataset. Indexing makes embedding requests the first time it sees a PDF or when its contents or model configuration changes. Later runs load `.data/vector_index.json`.

## Dataset format

The input dataset is JSON Lines with one object per case:

```json
{"question":"How do penguins swim?","expected_answer":"They propel themselves with their flippers."}
```

Generated records add `actual_answer` and `context`. Scored records add `answer_relevancy`, `faithfulness`, `context_relevancy`, and `correctness` scores.

## Evaluation notes

The metrics use an OpenRouter judge model and make external model requests. DeepEval is configured to run cases and metrics synchronously to avoid large concurrent credit reservations and cache successful metric results for reruns. A 402 response means OpenRouter does not have enough available credits for the request; use a lower-cost judge model or add credits. Evaluation scores are model judgments and should be reviewed alongside examples and source passages.

The index is a local JSON file and can be rebuilt with `uv run llm-evaluation-system index --force`. Generated data, local indexes, credentials, and DeepEval cache files are excluded from version control.

## Development

The installed entry point is `cli:main`; `python -m cli --help` shows commands. The evaluation dataset and sample PDF are kept in `datasets/` and `PDFs/` respectively.
