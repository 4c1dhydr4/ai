
import numpy as np

from threading import Lock
from django.conf import settings
from sentence_transformers import SentenceTransformer


MODEL_NAME = getattr(
    settings,
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DEVICE = getattr(settings, "RAG_EMBEDDING_DEVICE", "cpu")

EMBEDDING_DIMENSION = 384

_model = None
_model_lock = Lock()


def get_embedding_model():
    """
    Load the embedding model once per process.
    """
    global _model

    if _model is None:

        with _model_lock:

            if _model is None:

                _model = SentenceTransformer(
                    MODEL_NAME,
                    device=DEVICE,
                )

    return _model


def embed_texts(texts, batch_size=32):
    """
    Generate normalized embeddings for a list of texts.

    Returns:
        numpy.ndarray: float32 matrix (N, 384)
    """

    if not texts:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected embedding dimensions: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding contains invalid numeric values.")

    return embeddings


def embed_query(text):
    """
    Generate an embedding for a single query.
    """
    embeddings = embed_texts([text])

    return embeddings[0]
