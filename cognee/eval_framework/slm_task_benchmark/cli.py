"""Command-line interface: argument definitions and default resolution.

Kept separate from the runner so the runner reads as pure orchestration, mirroring the
layout of ``token_usage_analysis``.
"""

from __future__ import annotations

import argparse

from cognee.eval_framework.slm_task_benchmark.corpus import (
    CORPUS_FILES,
    DEFAULT_CORPUS,
    DEFAULT_MAX_CHUNK_SIZE,
)
from cognee.eval_framework.slm_task_benchmark.tasks import BENCHMARK_TASKS

DESCRIPTION = "Measure small local models against cognee's narrow LLM call sites."

# Both structured-output frameworks cognee ships that are relevant for local models.
# BAML is out of scope: it needs its own client registry and a separate provider config.
FRAMEWORKS = ("instructor", "litellm_native")

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate(args, parser)
    return args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slm_task_benchmark", description=DESCRIPTION)

    parser.add_argument(
        "--reference",
        action="store_true",
        help=(
            "Run the reference pass against the model cognee is configured with in .env "
            "and write results/reference.json. This is the only pass that calls a paid API."
        ),
    )
    parser.add_argument(
        "--models",
        type=_comma_list,
        default=[],
        help="Comma-separated local model ids to measure (e.g. qwen2.5:7b-instruct-q4_K_M).",
    )
    parser.add_argument(
        "--tasks",
        type=_comma_list,
        default=list(BENCHMARK_TASKS),
        help=f"Comma-separated task keys. Default: all. Available: {', '.join(BENCHMARK_TASKS)}",
    )
    parser.add_argument(
        "--frameworks",
        type=_comma_list,
        default=list(FRAMEWORKS),
        help=(
            "Structured-output frameworks to compare. Each runs as its own pass because "
            f"the setting is process-global. Available: {', '.join(FRAMEWORKS)}"
        ),
    )

    parser.add_argument("--samples", type=int, default=25, help="Samples per task (default 25).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42).")
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPUS_FILES),
        default=DEFAULT_CORPUS,
        help=f"Which bundled corpus to sample from (default {DEFAULT_CORPUS}).",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=DEFAULT_MAX_CHUNK_SIZE,
        help=f"Chunk size in tokens (default {DEFAULT_MAX_CHUNK_SIZE}).",
    )

    parser.add_argument(
        "--provider",
        default="ollama",
        help="cognee LLM provider used to reach the local models (default ollama).",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_OLLAMA_ENDPOINT,
        help=f"OpenAI-compatible endpoint of the local server (default {DEFAULT_OLLAMA_ENDPOINT}).",
    )
    parser.add_argument(
        "--api-key",
        default="ollama",
        help="API key for the local server. Local servers ignore it but the client needs one.",
    )

    parser.add_argument(
        "--label",
        default="run",
        help="Filename stem for the result file under results/ (default 'run').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-call timeout in seconds (default 180).",
    )
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.reference and not args.models:
        parser.error("Pass --models, or --reference to produce the reference outputs.")
    if args.reference and args.models:
        parser.error("--reference runs the configured default model; it takes no --models.")

    unknown_tasks = [key for key in args.tasks if key not in BENCHMARK_TASKS]
    if unknown_tasks:
        parser.error(
            f"Unknown task(s): {', '.join(unknown_tasks)}. Available: {', '.join(BENCHMARK_TASKS)}"
        )

    unknown_frameworks = [name for name in args.frameworks if name not in FRAMEWORKS]
    if unknown_frameworks:
        parser.error(
            f"Unknown framework(s): {', '.join(unknown_frameworks)}. "
            f"Available: {', '.join(FRAMEWORKS)}"
        )

    if args.samples < 1:
        parser.error("--samples must be at least 1.")


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
