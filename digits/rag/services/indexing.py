
from django.utils import timezone

from rag.models import (
    Document,
    DocumentIngestionLog,
)

from .embeddings import embed_texts, MODEL_NAME
from .vector_store import (
    add_vectors,
    get_document_partition,
)


def index_document(document):
    """
    Generate embeddings and index all document chunks.
    """

    document.embedding_status = Document.EmbeddingStatus.PROCESSING
    document.embedding_model = MODEL_NAME
    document.save()

    try:

        chunks = list(
            document.chunks.order_by("chunk_index")
        )

        if not chunks:
            raise ValueError(
                "Cannot index a document without chunks."
            )

        texts = [
            chunk.content
            for chunk in chunks
        ]

        # 1. Generate embeddings
        embeddings = embed_texts(
            texts,
            batch_size=32,
        )

        # 2. Get chunk IDs
        chunk_ids = [
            chunk.pk
            for chunk in chunks
        ]

        # 3. Determine the correct index partition
        partition = get_document_partition(document)

        # 4. Store vectors in FAISS
        add_vectors(
            partition=partition,
            chunk_ids=chunk_ids,
            vectors=embeddings,
        )

        # 5. Update document state
        document.embedding_status = Document.EmbeddingStatus.READY
        document.embedding_model = MODEL_NAME
        document.indexed_at = timezone.now()
        document.save()

        DocumentIngestionLog.objects.create(
            document=document,
            level=DocumentIngestionLog.Level.INFO,
            message="Document embeddings generated and indexed.",
            details={
                "model": MODEL_NAME,
                "dimensions": embeddings.shape[1],
                "vectors": len(embeddings),
                "partition": partition,
            },
        )

        return document

    except Exception as exc:

        document.embedding_status = Document.EmbeddingStatus.FAILED
        document.save()

        DocumentIngestionLog.objects.create(
            document=document,
            level=DocumentIngestionLog.Level.ERROR,
            message="Embedding generation or indexing failed.",
            details={
                "error": str(exc),
                "exception_type": type(exc).__name__,
            },
        )

        raise
