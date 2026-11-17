
from django.db.models import Q

from rag.models import Document, DocumentChunk
from .embeddings import embed_query
from .vector_store import get_partition_key, search_vectors


DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.25


def retrieve_context(
    query,
    user,
    conversation=None,
    top_k=DEFAULT_TOP_K,
    min_score=DEFAULT_MIN_SCORE,
):
    """
    Retrieve relevant document chunks from:
      - The user's global knowledge base.
      - The active conversation's knowledge base.

    Returns:
        {
            "context": str,
            "sources": list,
            "results": list,
        }
    """

    query = (query or "").strip()

    if not query:
        return {
            "context": "",
            "sources": [],
            "results": [],
        }

    if top_k < 1:
        raise ValueError("top_k must be greater than zero.")

    if conversation is not None:
        if conversation.user_id != user.pk:
            raise PermissionError(
                "The conversation does not belong to this user."
            )

    # 1. Embed the question once.
    query_vector = embed_query(query)

    # 2. Search the global knowledge base.
    global_partition = get_partition_key(
        user_id=user.pk,
        scope=Document.Scope.GLOBAL,
    )

    candidates = search_vectors(
        partition=global_partition,
        query_vector=query_vector,
        top_k=top_k * 3,
    )

    # 3. Search the active conversation's knowledge base.
    if conversation is not None:
        conversation_partition = get_partition_key(
            user_id=user.pk,
            scope=Document.Scope.CONVERSATION,
            conversation_id=conversation.pk,
        )

        candidates.extend(
            search_vectors(
                partition=conversation_partition,
                query_vector=query_vector,
                top_k=top_k * 3,
            )
        )

    if not candidates:
        return {
            "context": "",
            "sources": [],
            "results": [],
        }

    # 4. Merge candidates and keep the best score for each chunk.
    ranked_candidates = {}

    for candidate in candidates:
        chunk_id = candidate["chunk_id"]
        score = candidate["score"]

        if score < min_score:
            continue

        if (
            chunk_id not in ranked_candidates
            or score > ranked_candidates[chunk_id]["score"]
        ):
            ranked_candidates[chunk_id] = candidate

    if not ranked_candidates:
        return {
            "context": "",
            "sources": [],
            "results": [],
        }

    # 5. Retrieve canonical records from PostgreSQL.
    chunk_ids = list(ranked_candidates.keys())

    valid_documents = Q(
        document__scope=Document.Scope.GLOBAL,
        document__conversation__isnull=True,
    )

    if conversation is not None:
        valid_documents |= Q(
            document__scope=Document.Scope.CONVERSATION,
            document__conversation_id=conversation.pk,
        )

    chunks = (
        DocumentChunk.objects
        .filter(
            Q(
                document__user=user,
                document__status=Document.Status.READY,
                document__embedding_status=Document.EmbeddingStatus.READY,
            )
            & valid_documents,
            pk__in=chunk_ids,
        )
        .select_related("document")
    )

    chunks_by_id = {
        chunk.pk: chunk
        for chunk in chunks
    }

    # 6. Restore similarity ordering and discard stale/invalid IDs.
    results = []

    for chunk_id, candidate in ranked_candidates.items():
        chunk = chunks_by_id.get(chunk_id)

        if chunk is None:
            continue

        results.append({
            "chunk": chunk,
            "score": candidate["score"],
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    results = results[:top_k]

    # 7. Build source metadata and formatted context.
    sources = []
    context_parts = []

    for index, result in enumerate(results, start=1):
        chunk = result["chunk"]
        document = chunk.document
        metadata = chunk.metadata or {}

        source = {
            "source_id": index,
            "document_id": document.pk,
            "title": document.title,
            "scope": document.scope,
            "chunk_id": chunk.pk,
            "chunk_index": chunk.chunk_index,
            "page": metadata.get("page"),
            "score": round(result["score"], 4),
        }

        sources.append(source)

        location = (
            f", page {source['page']}"
            if source["page"] is not None
            else ""
        )

        context_parts.append(
            f"[Source {index}: {document.title}{location}]\n"
            f"{chunk.content}"
        )

    return {
        "context": "\n\n---\n\n".join(context_parts),
        "sources": sources,
        "results": results,
    }
