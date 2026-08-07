from typing import List, Union, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity
from cognee.tasks.temporal_graph.models import Event


class DocumentChunk(DataPoint):
    """
    Represents a chunk of text from a document with associated metadata.

    Public methods include:

    - No public methods defined in the provided code.

    Instance variables include:

    - text: The textual content of the chunk.
    - chunk_size: The size of the chunk.
    - chunk_index: The index of the chunk in the original document.
    - cut_type: The type of cut that defined this chunk.
    - is_part_of: The document to which this chunk belongs.
    - contains: A list of entities or events contained within the chunk (default is None).
    - document_id: Flat string id of the source document, for reference rendering.
    - document_name: Display name (basename) of the source document, for reference rendering.
    - page_start: 1-based page (or slide) the chunk starts on, when the source format allows
    it to be derived. None otherwise.
    - page_end: 1-based page the chunk ends on. Equal to page_start unless the chunk spans a
    page break.
    - metadata: A dictionary to hold meta information related to the chunk, including index
    fields.
    """

    text: str
    chunk_size: int
    chunk_index: int
    cut_type: str
    is_part_of: Document
    contains: List[Union[Entity, Event, tuple[Edge, Entity]]] = None
    importance_weight: Optional[float] = 0.5
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    # Optional truth-alignment fields; never embedded (kept out of index_fields)
    # and not part of id/dedup.
    truth_alignment: Optional[list[float]] = None
    truth_epoch: Optional[int] = None
    # Optional page provenance; never embedded (kept out of index_fields). Both stay None
    # when the source has no derivable pagination — see chunking/page_markers.py.
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: dict = {"index_fields": ["text"]}
