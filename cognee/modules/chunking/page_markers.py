"""Recover page numbers from the ``Page N:`` markers loaders leave in the text.

Loaders that know about pagination (``PyPdfLoader``, ``AdvancedPdfLoader``, and
``UnstructuredLoader`` for docx/pptx/xlsx) prefix each page's text with a
``Page N:`` line. That marker is ordinary text inside the document, not a
structured field, so a chunk's page range has to be recovered by reading the
chunk's own text.

Two things make that more than a regex match:

- A chunk often contains no marker at all, because the marker landed in an
  earlier chunk. Such a chunk still belongs to the last page seen, so the page
  has to be carried forward across chunks.
- A chunk can straddle a page break, in which case it has a *range* rather than
  a single page.

``PageTracker`` holds the carried-forward page so callers cannot forget to
thread it through. Chunkers process a document's chunks in order, one tracker
per document.
"""

import re
from typing import Optional, Tuple

# The literal the loaders emit: "Page 12:" alone on its line. Multiline mode so
# ^ and $ anchor to line boundaries inside the chunk, not just its ends.
# AdvancedPdfLoader emits a bare "Page:" when it has no number; that carries no
# information and is deliberately not matched.
_PAGE_MARKER = re.compile(r"(?m)^Page (\d+):$")


class PageTracker:
    """Assigns a page range to each chunk of one document, in document order.

    Create one per document and call :meth:`stamp` with each chunk's text in the
    order the chunks appear. Returns ``(None, None)`` for every chunk until the
    first marker is seen — a source without pagination never gets a fabricated
    page number.
    """

    def __init__(self) -> None:
        self._current_page: Optional[int] = None

    def stamp(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Return ``(page_start, page_end)`` for one chunk and advance the tracker.

        Parameters
        ----------
        text:
            The chunk's own text, in original document order.
        """
        markers = list(_PAGE_MARKER.finditer(text))

        if not markers:
            # No marker here: this chunk sits inside whatever page was last seen.
            return self._current_page, self._current_page

        first, last = markers[0], markers[-1]
        # Text before the first marker is the tail of the previous page, so the
        # chunk starts there. A chunk that begins at the marker starts on the
        # marker's own page instead.
        has_leading_content = bool(text[: first.start()].strip())
        page_start = (
            self._current_page
            if (self._current_page is not None and has_leading_content)
            else int(first.group(1))
        )
        page_end = int(last.group(1))

        self._current_page = page_end
        return page_start, page_end
