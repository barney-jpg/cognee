"""Unit tests asserting the chunkers stamp page provenance onto DocumentChunk.

Runs the real chunkers over an in-memory text generator shaped like what the PDF
loaders produce (no loader, no LLM, no network), and checks that page_start and
page_end land on every chunk — including the ones that carry no marker of their
own. PageTracker's own logic is covered in test_page_markers.py.
"""

from uuid import uuid4

import pytest

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.text_chunker_with_overlap import TextChunkerWithOverlap
from cognee.modules.data.processing.document_types import Document


@pytest.fixture(params=["TextChunker", "TextChunkerWithOverlap"])
def chunker_class(request):
    return TextChunker if request.param == "TextChunker" else TextChunkerWithOverlap


def _make_text_generator(*texts):
    async def gen():
        for text in texts:
            yield text

    return gen


async def _collect(chunker):
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


def _document() -> Document:
    return Document(
        id=uuid4(),
        name="report.pdf",
        raw_data_location="/abs/path/to/report.pdf",
        external_metadata=None,
        mime_type="text/plain",
    )


def _paged_text(pages: int, sentences_per_page: int = 12) -> str:
    """Mimic PyPdfLoader output: "Page N:" then the page body, joined by newlines."""
    blocks = [
        f"Page {number}:\n"
        + " ".join(f"Sentence {i} of page {number}." for i in range(sentences_per_page))
        + "\n"
        for number in range(1, pages + 1)
    ]
    return "\n".join(blocks)


@pytest.mark.asyncio
async def test_every_chunk_of_a_paged_document_carries_a_page(chunker_class):
    """Chunks without a marker of their own must inherit the carried page."""
    chunker = chunker_class(_document(), _make_text_generator(_paged_text(4)), max_chunk_size=64)

    chunks = await _collect(chunker)

    assert len(chunks) > 4, "expected the pages to split into more chunks than pages"
    for chunk in chunks:
        assert chunk.page_start is not None
        assert chunk.page_end is not None
        assert chunk.page_start <= chunk.page_end


@pytest.mark.asyncio
async def test_pages_are_assigned_in_document_order(chunker_class):
    """Page numbers must never go backwards as chunks progress."""
    chunker = chunker_class(_document(), _make_text_generator(_paged_text(4)), max_chunk_size=64)

    chunks = await _collect(chunker)

    page_starts = [chunk.page_start for chunk in chunks]
    assert page_starts == sorted(page_starts)
    assert page_starts[0] == 1
    assert chunks[-1].page_end == 4


@pytest.mark.asyncio
async def test_a_document_without_markers_leaves_pages_unset(chunker_class):
    """Pasted text has no pages; the fields stay None rather than defaulting to 1."""
    text = " ".join(f"Sentence {i} with no page marker anywhere." for i in range(30))
    chunker = chunker_class(_document(), _make_text_generator(text), max_chunk_size=64)

    chunks = await _collect(chunker)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.page_start is None
        assert chunk.page_end is None


@pytest.mark.asyncio
async def test_page_fields_are_not_embedded(chunker_class):
    """Page numbers are provenance, not meaning: they must stay out of index_fields."""
    chunker = chunker_class(_document(), _make_text_generator(_paged_text(2)), max_chunk_size=64)

    chunks = await _collect(chunker)

    for chunk in chunks:
        assert chunk.metadata["index_fields"] == ["text"]
