"""Command-line interface for the RAG and evaluation workflows."""

from __future__ import annotations

import argparse

from evaluation.generation import generate_predictions
from evaluation.scoring import score_predictions
from rag.agent import answer_question
from rag.indexing import build_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-evaluation-system",
        description="Run a PDF RAG pipeline and evaluate its answers.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="Ask a question about the configured PDFs")
    ask.add_argument("question", help="Question to answer")
    ask.add_argument("--model", help="OpenRouter chat model ID")

    index = commands.add_parser("index", help="Build or refresh the persistent PDF vector index")
    index.add_argument("--source", help="PDF file or directory (defaults to RAG_PDF_PATH)")
    index.add_argument("--force", action="store_true", help="Re-embed documents even when the cache is valid")

    generate = commands.add_parser("generate", help="Generate RAG answers for a JSONL dataset")
    generate.add_argument("--dataset", default="datasets/eval_dataset.jsonl")
    generate.add_argument("--output", default="data/predictions.jsonl")
    generate.add_argument("--limit", type=int)

    score = commands.add_parser("score", help="Score saved predictions with DeepEval")
    score.add_argument("--input", default="data/predictions.jsonl")
    score.add_argument("--output", default="data/evaluation_results.jsonl")
    score.add_argument("--model", help="OpenRouter judge model ID")
    score.add_argument("--limit", type=int)

    run = commands.add_parser("evaluate", help="Generate answers and then score them")
    run.add_argument("--dataset", default="datasets/eval_dataset.jsonl")
    run.add_argument("--predictions", default="data/predictions.jsonl")
    run.add_argument("--output", default="data/evaluation_results.jsonl")
    run.add_argument("--chat-model", help="OpenRouter answer model ID")
    run.add_argument("--judge-model", help="OpenRouter evaluation model ID")
    run.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "ask":
            result = answer_question(args.question, model=args.model)
            print(result.answer)
            print("\nRetrieved context:\n" + result.context)
        elif args.command == "index":
            store = build_index(args.source, force_rebuild=args.force)
            print(f"Vector index is ready with {len(store.store)} chunks.")
        elif args.command == "generate":
            output = generate_predictions(args.dataset, args.output, limit=args.limit)
            print(f"Predictions saved to {output}")
        elif args.command == "score":
            output = score_predictions(
                args.input, args.output, model=args.model, limit=args.limit
            )
            print(f"Evaluation results saved to {output}")
        elif args.command == "evaluate":
            predictions = generate_predictions(
                args.dataset,
                args.predictions,
                limit=args.limit,
                model=args.chat_model,
            )
            output = score_predictions(
                predictions,
                args.output,
                model=args.judge_model,
                limit=args.limit,
            )
            print(f"Evaluation results saved to {output}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
