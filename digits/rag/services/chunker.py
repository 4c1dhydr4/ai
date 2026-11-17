
def split_text(text, chunk_size=1000, overlap=150):
    """
    Split text into overlapping chunks without
    unnecessarily cutting words.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_sections(sections, chunk_size=1000, overlap=150):
    """
    Chunk extracted sections and preserve their metadata.
    """
    results = []

    for section in sections:

        chunks = split_text(
            section["content"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            results.append({
                "content": chunk,
                "metadata": section.get("metadata", {}).copy(),
            })

    return results
