"""Sampled inputs for the benchmark: text chunks, chunk pairs and search queries.

Chunking is delegated to ``token_usage_analysis/corpus.py``, which runs cognee's own
``TextChunker`` against a local tiktoken tokenizer. That module deliberately patches the
embedding engine away, so producing inputs needs no LLM call and no network access.
Reusing it keeps the chunk-length distribution identical to what ingestion actually sees.
"""

from __future__ import annotations

import random
from pathlib import Path

from cognee.eval_framework.token_usage_analysis.corpus import chunk_text, sample_chunks

# Corpora shipped with token_usage_analysis, spanning a range of entity density.
_DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "token_usage_analysis" / "data"

CORPUS_FILES = {
    "wikipedia": "wikipedia_article.txt",
    "wikinews": "wikinews_article.txt",
    "fiction": "war_and_peace_excerpt.txt",
    "dense": "dense_synthetic.txt",
}

DEFAULT_CORPUS = "wikipedia"

# Chunk size is capped well below cognee's 8191 default: a 7B model at 4k context has to
# fit system prompt + schema + chunk, and an over-long chunk would measure truncation
# rather than schema adherence.
DEFAULT_MAX_CHUNK_SIZE = 1024

# Free-form queries for the search-type routing task. Deliberately mixed in shape —
# factual lookup, relationship questions, summarization requests, time-scoped questions —
# so a router has genuinely different options to choose between.
BENCHMARK_QUERIES = [
    "What is the Apollo 11 mission?",
    "Who was involved in the moon landing?",
    "Summarize the main events described in the documents.",
    "How are Neil Armstrong and Buzz Aldrin related to each other?",
    "List every organization mentioned in the corpus.",
    "What happened in July 1969?",
    "Give me a short overview of the whole dataset.",
    "Which people worked together on the same project?",
    "What does the text say about the lunar module?",
    "Find passages that mention radio communication.",
    "Why did the mission succeed?",
    "What were the main risks discussed?",
    "Show me the connection between NASA and the astronauts.",
    "When did the spacecraft return to Earth?",
    "Explain the sequence of events step by step.",
    "What are the key facts I should know?",
    "Which locations are referenced in the documents?",
    "Compare the roles of the crew members.",
    "What technical systems are described?",
    "Give me the exact wording about the first step on the moon.",
    "How did public reaction develop over time?",
    "What preceded the launch?",
    "Who reported on these events?",
    "Summarize what each person did.",
    "What is the relationship between the command module and the lunar module?",
    "Are there any references to earlier missions?",
    "What happened after the landing?",
    "Which dates are mentioned in the corpus?",
    "Describe the overall narrative in two sentences.",
    "What equipment was carried on board?",
]


def load_corpus_text(corpus: str = DEFAULT_CORPUS) -> str:
    """Read one of the bundled corpora by short name."""
    if corpus not in CORPUS_FILES:
        raise ValueError(f"Unknown corpus {corpus!r}. Available: {', '.join(sorted(CORPUS_FILES))}")
    return (_DATA_DIRECTORY / CORPUS_FILES[corpus]).read_text(encoding="utf-8")


def sampled_chunks(
    corpus: str = DEFAULT_CORPUS,
    samples: int = 25,
    seed: int = 42,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[str]:
    """Chunk a corpus with cognee's chunker, then take a deterministic sample."""
    chunks = chunk_text(load_corpus_text(corpus), max_chunk_size=max_chunk_size)
    return sample_chunks(chunks, sample_size=samples, seed=seed)


def sampled_chunk_pairs(
    corpus: str = DEFAULT_CORPUS,
    samples: int = 25,
    seed: int = 42,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[tuple[str, str]]:
    """Deterministic chunk pairs for the association task.

    Pairs mix near neighbours with distant chunks, so the task sees both plausibly
    related and plausibly unrelated input rather than only one of the two.
    """
    chunks = chunk_text(load_corpus_text(corpus), max_chunk_size=max_chunk_size)
    if len(chunks) < 2:
        raise ValueError(f"Corpus {corpus!r} produced fewer than two chunks; cannot form pairs.")

    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for index in range(samples):
        first = rng.randrange(len(chunks))
        # Alternate between an adjacent chunk (likely related) and a random one.
        if index % 2 == 0 and first + 1 < len(chunks):
            second = first + 1
        else:
            second = rng.randrange(len(chunks))
            while second == first:
                second = rng.randrange(len(chunks))
        pairs.append((chunks[first], chunks[second]))
    return pairs


def sampled_queries(samples: int = 25, seed: int = 42) -> list[str]:
    """Deterministic sample of the built-in search queries."""
    if samples >= len(BENCHMARK_QUERIES):
        return list(BENCHMARK_QUERIES)
    return random.Random(seed).sample(BENCHMARK_QUERIES, samples)
