import re


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks.

    Strategy: split on paragraph boundaries, then sentences if paragraphs
    are too large, with overlap between chunks.
    """
    if not text or not text.strip():
        return []

    paragraphs = re.split(r"\n\n+", text.strip())
    segments: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            segments.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                segments.append(sentence.strip())

    chunks: list[str] = []
    current = ""

    for segment in segments:
        if current and len(current) + len(segment) + 1 > chunk_size:
            chunks.append(current.strip())
            if chunk_overlap > 0 and len(current) > chunk_overlap:
                current = current[-chunk_overlap:] + " " + segment
            else:
                current = segment
        else:
            current = current + " " + segment if current else segment

    if current.strip():
        chunks.append(current.strip())

    return chunks
