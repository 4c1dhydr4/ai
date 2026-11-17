
from django.db import transaction
from django.utils import timezone

from rag.models import (
    Document,
    DocumentChunk,
    DocumentIngestionLog,
)

from .document_loader import extract_document
from .chunker import chunk_sections
from .vector_store import remove_document_vectors


def ingest_document(document):
    """
    Extract, chunk, and persist a document.

    Pipeline:
        Document -> Extraction -> Chunking -> Database
    """

    document.status = Document.Status.PROCESSING
    document.error_message = ""
    document.processed_at = None
    document.save()

    try:

        # 1. Extract document contents
        sections = extract_document(document)

        if not sections:
            raise ValueError(
                "No readable text was found in the document."
            )

        # 2. Generate text chunks
        chunks = chunk_sections(
            sections,
            chunk_size=1000,
            overlap=150,
        )

        if not chunks:
            raise ValueError(
                "Document extraction produced no chunks."
            )

        # 3. Persist chunks and update document status
        with transaction.atomic():

            # Remove previous chunks if reprocessing
            document.chunks.all().delete()
            remove_document_vectors(document)

            DocumentChunk.objects.bulk_create([
                DocumentChunk(
                    document=document,
                    content=chunk["content"],
                    chunk_index=index,
                    metadata=chunk["metadata"],
                )
                for index, chunk in enumerate(chunks)
            ])

            document.status = Document.Status.READY
            document.processed_at = timezone.now()
            document.error_message = ""
            document.save()

            DocumentIngestionLog.objects.create(
                document=document,
                level=DocumentIngestionLog.Level.INFO,
                message="Document successfully ingested.",
                details={
                    "sections": len(sections),
                    "chunks": len(chunks),
                    "chunk_size": 1000,
                    "overlap": 150,
                },
            )

        return document

    except Exception as exc:

        document.status = Document.Status.FAILED
        document.error_message = str(exc)
        document.save()

        DocumentIngestionLog.objects.create(
            document=document,
            level=DocumentIngestionLog.Level.ERROR,
            message="Document ingestion failed.",
            details={
                "error": str(exc),
                "exception_type": type(exc).__name__,
            },
        )

        raise
