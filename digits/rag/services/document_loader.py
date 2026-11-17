
from io import BytesIO
from pathlib import Path

import fitz
from docx import Document as WordDocument


def extract_pdf(file_bytes):
    """
    Extract text from PDF, preserving page numbers.
    """
    sections = []

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:
                sections.append({
                    "content": text,
                    "metadata": {
                        "page": page_number,
                        "source_type": "pdf",
                    }
                })

    return sections


def extract_docx(file_bytes):
    """
    Extract paragraphs from DOCX documents.
    """
    document = WordDocument(BytesIO(file_bytes))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    text = "\n\n".join(paragraphs)

    if not text:
        return []

    return [{
        "content": text,
        "metadata": {
            "source_type": "docx",
        }
    }]


def extract_text(file_bytes):
    """
    Extract UTF-8 text from TXT and Markdown files.
    """
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("utf-8", errors="replace")

    text = text.strip()

    if not text:
        return []

    return [{
        "content": text,
        "metadata": {
            "source_type": "text",
        }
    }]


def extract_document(document):
    """
    Dispatch extraction based on the document file type.
    """
    extension = Path(document.file.name).suffix.lower()

    document.file.open("rb")

    try:
        file_bytes = document.file.read()
    finally:
        document.file.close()

    if extension == ".pdf":
        return extract_pdf(file_bytes)

    if extension == ".docx":
        return extract_docx(file_bytes)

    if extension in (".txt", ".md"):
        return extract_text(file_bytes)

    raise ValueError(
        f"Unsupported document format: {extension}"
    )
