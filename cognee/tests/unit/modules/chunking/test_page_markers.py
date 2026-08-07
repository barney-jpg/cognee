"""Unit tests for PageTracker, which recovers page ranges from "Page N:" markers.

Pure logic over strings: no loader, no chunker, no I/O. The chunker-side wiring
is covered separately in test_chunk_page_fields.py.
"""

from cognee.modules.chunking.page_markers import PageTracker


class TestWithoutMarkers:
    def test_a_source_with_no_pagination_never_gets_a_page(self):
        """Pasted text and CSV rows must stay unpaged rather than get a made-up 1."""
        tracker = PageTracker()

        assert tracker.stamp("Some text with no marker at all.") == (None, None)
        assert tracker.stamp("More text.") == (None, None)

    def test_a_lookalike_that_is_not_a_marker_is_ignored(self):
        """Only the loaders' exact on-its-own-line form counts."""
        tracker = PageTracker()

        assert tracker.stamp("See Page 4: for details, as noted on page 7.") == (None, None)

    def test_a_bare_page_marker_without_a_number_is_ignored(self):
        """AdvancedPdfLoader emits 'Page:' when it has no number; it carries nothing."""
        tracker = PageTracker()

        assert tracker.stamp("Page:\nSome text.") == (None, None)


class TestSingleMarker:
    def test_a_chunk_starting_at_a_marker_takes_that_page(self):
        tracker = PageTracker()

        assert tracker.stamp("Page 1:\nOpening paragraph.") == (1, 1)

    def test_a_chunk_after_a_marker_inherits_the_carried_page(self):
        """The common case: the marker landed in an earlier chunk."""
        tracker = PageTracker()
        tracker.stamp("Page 3:\nStart of page three.")

        assert tracker.stamp("Continuation with no marker.") == (3, 3)
        assert tracker.stamp("Still page three.") == (3, 3)


class TestPageBreaks:
    def test_a_chunk_spanning_a_break_reports_a_range(self):
        """Content before the marker belongs to the previous page."""
        tracker = PageTracker()
        tracker.stamp("Page 1:\nFirst page.")

        assert tracker.stamp("Tail of page one.\nPage 2:\nHead of page two.") == (1, 2)

    def test_a_chunk_covering_several_breaks_spans_first_to_last(self):
        tracker = PageTracker()
        tracker.stamp("Page 1:\nFirst.")

        assert tracker.stamp("tail\nPage 2:\nmid\nPage 3:\nend") == (1, 3)

    def test_the_carried_page_advances_to_the_last_marker_seen(self):
        tracker = PageTracker()
        tracker.stamp("Page 1:\ntext\nPage 2:\ntext")

        assert tracker.stamp("no marker here") == (2, 2)

    def test_a_chunk_that_only_repeats_the_marker_does_not_inherit_backwards(self):
        """A chunk beginning exactly at a marker starts on that marker's page.

        Overlapping chunkers re-emit the marker at the head of the next chunk; that
        chunk belongs to the new page, not the previous one.
        """
        tracker = PageTracker()
        tracker.stamp("Page 4:\nbody")

        assert tracker.stamp("Page 5:\nbody") == (5, 5)


class TestIndependence:
    def test_two_trackers_do_not_share_state(self):
        """One tracker per document — a second document restarts unpaged."""
        first = PageTracker()
        first.stamp("Page 9:\ntext")

        second = PageTracker()
        assert second.stamp("text with no marker") == (None, None)
