from cognee.shared.logging_utils import get_logger
from os.path import basename
from uuid import NAMESPACE_OID, uuid5

from cognee.tasks.chunks import chunk_by_paragraph
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.page_markers import PageTracker
from .models.DocumentChunk import DocumentChunk

logger = get_logger()


class TextChunker(Chunker):
    async def read(self):
        document_id = str(self.document.id)
        document_name = self.document.name or basename(self.document.raw_data_location)
        paragraph_chunks = []
        # One tracker per document: it carries the current page across chunks that
        # contain no marker of their own.
        page_tracker = PageTracker()
        async for content_text in self.get_text():
            for chunk_data in chunk_by_paragraph(
                content_text,
                self.max_chunk_size,
                batch_paragraphs=True,
            ):
                if self.chunk_size + chunk_data["chunk_size"] <= self.max_chunk_size:
                    paragraph_chunks.append(chunk_data)
                    self.chunk_size += chunk_data["chunk_size"]
                else:
                    if len(paragraph_chunks) == 0:
                        page_start, page_end = page_tracker.stamp(chunk_data["text"])
                        yield DocumentChunk(
                            id=chunk_data["chunk_id"],
                            text=chunk_data["text"],
                            chunk_size=chunk_data["chunk_size"],
                            is_part_of=self.document,
                            chunk_index=self.chunk_index,
                            cut_type=chunk_data["cut_type"],
                            contains=[],
                            importance_weight=self.document.importance_weight,
                            document_id=document_id,
                            document_name=document_name,
                            page_start=page_start,
                            page_end=page_end,
                            metadata={
                                "index_fields": ["text"],
                            },
                        )
                        paragraph_chunks = []
                        self.chunk_size = 0
                    else:
                        chunk_text = " ".join(chunk["text"] for chunk in paragraph_chunks)
                        page_start, page_end = page_tracker.stamp(chunk_text)
                        try:
                            yield DocumentChunk(
                                id=uuid5(
                                    NAMESPACE_OID, f"{str(self.document.id)}-{self.chunk_index}"
                                ),
                                text=chunk_text,
                                chunk_size=self.chunk_size,
                                is_part_of=self.document,
                                chunk_index=self.chunk_index,
                                cut_type=paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                                contains=[],
                                importance_weight=self.document.importance_weight,
                                document_id=document_id,
                                document_name=document_name,
                                page_start=page_start,
                                page_end=page_end,
                                metadata={
                                    "index_fields": ["text"],
                                },
                            )
                        except Exception as e:
                            logger.error(e)
                            raise e
                        paragraph_chunks = [chunk_data]
                        self.chunk_size = chunk_data["chunk_size"]

                    self.chunk_index += 1

        if len(paragraph_chunks) > 0:
            final_text = " ".join(chunk["text"] for chunk in paragraph_chunks)
            page_start, page_end = page_tracker.stamp(final_text)
            try:
                yield DocumentChunk(
                    id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{self.chunk_index}"),
                    text=final_text,
                    chunk_size=self.chunk_size,
                    is_part_of=self.document,
                    chunk_index=self.chunk_index,
                    cut_type=paragraph_chunks[len(paragraph_chunks) - 1]["cut_type"],
                    contains=[],
                    importance_weight=self.document.importance_weight,
                    document_id=document_id,
                    document_name=document_name,
                    page_start=page_start,
                    page_end=page_end,
                    metadata={"index_fields": ["text"]},
                )
            except Exception as e:
                logger.error(e)
                raise e
