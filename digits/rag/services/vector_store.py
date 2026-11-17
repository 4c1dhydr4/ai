
import hashlib
import os

from pathlib import Path

import faiss
import numpy as np

from filelock import FileLock
from django.conf import settings


DIMENSION = settings.RAG_EMBEDDING_DIMENSION


def get_partition_key(user_id, scope, conversation_id=None):
    """
    Generate an isolated index partition.
    """

    if scope == "global":
        return f"user:{user_id}:global"

    if scope == "conversation" and conversation_id is not None:
        return f"user:{user_id}:conversation:{conversation_id}"

    raise ValueError("Invalid document index partition.")


def get_document_partition(document):
    return get_partition_key(
        user_id=document.user_id,
        scope=document.scope,
        conversation_id=document.conversation_id,
    )


def get_index_paths(partition):
    """
    Generate safe, deterministic paths for index and lock files.
    """

    root = Path(settings.RAG_FAISS_INDEX_DIR)
    root.mkdir(parents=True, exist_ok=True)

    partition_hash = hashlib.sha256(
        partition.encode("utf-8")
    ).hexdigest()

    return (
        root / f"{partition_hash}.faiss",
        root / f"{partition_hash}.lock",
    )


def create_index():
    """
    Create an exact cosine-similarity index.
    """

    base_index = faiss.IndexFlatIP(DIMENSION)

    return faiss.IndexIDMap2(base_index)


def load_index(index_path):
    """
    Load an existing FAISS index or create a new one.
    """

    if not index_path.exists():
        return create_index()

    index = faiss.read_index(str(index_path))

    if index.d != DIMENSION:
        raise ValueError(
            f"Index dimension mismatch: {index.d} != {DIMENSION}"
        )

    return index


def save_index(index, index_path):
    """
    Persist an index atomically.
    """

    temporary_path = index_path.with_name(
        f"{index_path.stem}.tmp.faiss"
    )

    try:
        faiss.write_index(index, str(temporary_path))
        os.replace(temporary_path, index_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def add_vectors(partition, chunk_ids, vectors):
    """
    Insert or update vectors in a FAISS partition.

    Each vector is identified by its DocumentChunk primary key.
    """

    chunk_ids = np.asarray(
        chunk_ids,
        dtype=np.int64,
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION:
        raise ValueError("Invalid vector dimensions.")

    if len(chunk_ids) != len(vectors):
        raise ValueError("Chunk IDs and vectors have different lengths.")

    if not len(chunk_ids):
        return

    if len(np.unique(chunk_ids)) != len(chunk_ids):
        raise ValueError("Duplicate chunk IDs are not allowed.")

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors contain invalid numeric values.")

    if not np.allclose(
        np.linalg.norm(vectors, axis=1),
        1.0,
        atol=1e-3,
    ):
        raise ValueError("Vectors must be normalized.")

    index_path, lock_path = get_index_paths(partition)

    with FileLock(str(lock_path)):

        index = load_index(index_path)

        # Remove existing IDs before updating them.
        index.remove_ids(chunk_ids)

        index.add_with_ids(
            vectors,
            chunk_ids,
        )

        save_index(index, index_path)


def search_vectors(partition, query_vector, top_k=5):
    """
    Search a partition using cosine similarity.

    Returns:
        [
            {"chunk_id": 123, "score": 0.91},
            ...
        ]
    """

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32,
    ).reshape(1, -1)

    if query_vector.shape[1] != DIMENSION:
        raise ValueError("Invalid query vector dimensions.")

    if not np.isfinite(query_vector).all():
        raise ValueError("Query vector contains invalid values.")

    norm = np.linalg.norm(query_vector)

    if norm == 0:
        raise ValueError("Query vector cannot be zero.")

    query_vector /= norm

    if top_k <= 0:
        return []

    index_path, lock_path = get_index_paths(partition)

    if not index_path.exists():
        return []

    with FileLock(str(lock_path)):

        index = faiss.read_index(str(index_path))

        if index.d != DIMENSION:
            raise ValueError("FAISS index dimension mismatch.")

        if index.ntotal == 0:
            return []

        scores, ids = index.search(
            query_vector,
            min(top_k, index.ntotal),
        )

    results = []

    for chunk_id, score in zip(ids[0], scores[0]):

        if chunk_id < 0:
            continue

        results.append({
            "chunk_id": int(chunk_id),
            "score": float(score),
        })

    return results


def remove_vectors(partition, chunk_ids):
    """
    Remove vectors by their DocumentChunk primary keys.
    """

    chunk_ids = np.asarray(
        chunk_ids,
        dtype=np.int64,
    )

    if not len(chunk_ids):
        return

    index_path, lock_path = get_index_paths(partition)

    if not index_path.exists():
        return

    with FileLock(str(lock_path)):

        index = load_index(index_path)

        index.remove_ids(chunk_ids)

        save_index(index, index_path)


def remove_document_vectors(document):
    """
    Remove every indexed chunk belonging to a document.
    """

    chunk_ids = list(
        document.chunks.values_list("pk", flat=True)
    )

    if not chunk_ids:
        return

    partition = get_document_partition(document)

    remove_vectors(partition, chunk_ids)
