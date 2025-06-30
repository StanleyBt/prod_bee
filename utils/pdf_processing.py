import logging
import fitz  # PyMuPDF
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from the PDF file at pdf_path.
    """
    logger.info(f"Opening PDF file: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for i, page in enumerate(doc):
        page_text = page.get_text()
        logger.debug(f"Extracted text from page {i+1}: {len(page_text)} characters")
        full_text += page_text
    logger.info(f"Total extracted text length: {len(full_text)} characters")
    return full_text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into chunks of chunk_size words with overlap.
    """
    words = text.split()
    logger.info(f"Splitting text into chunks with chunk size {chunk_size} and overlap {overlap}")
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        logger.debug(f"Created chunk {len(chunks)} with {end - start} words")
        start += chunk_size - overlap
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks

def extract_chunks_from_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Extract text from a PDF file and split into chunks.
    """
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text, chunk_size, overlap)
