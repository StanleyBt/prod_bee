import logging
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from core.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SINGLE_CHUNK_PER_DOCUMENT

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract all text and metadata from the PDF file at pdf_path.
    Returns a dictionary with text and metadata.
    """
    logger.info(f"Opening PDF file: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    # Extract metadata
    metadata = doc.metadata
    page_count = len(doc)
    
    # Extract text
    full_text = ""
    page_texts = []
    for i, page in enumerate(doc):
        page_text = page.get_text()
        page_texts.append(page_text)
        logger.debug(f"Extracted text from page {i+1}: {len(page_text)} characters")
        full_text += page_text
    
    # Calculate additional metadata
    word_count = len(full_text.split())
    char_count = len(full_text)
    
    result = {
        "text": full_text,
        "metadata": {
            "source": str(pdf_path),
            "filename": Path(pdf_path).name,
            "page_count": page_count,
            "word_count": word_count,
            "char_count": char_count,
            "extraction_date": datetime.now().isoformat(),
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", "")
        }
    }
    
    logger.info(f"Total extracted text length: {char_count} characters, {word_count} words, {page_count} pages")
    return result

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
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

def extract_chunks_from_pdf(pdf_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
    """
    Extract text from a PDF file and create chunks with metadata.
    Returns a list of dictionaries containing chunk content and metadata.
    """
    pdf_data = extract_text_from_pdf(pdf_path)
    text = pdf_data["text"]
    metadata = pdf_data["metadata"]
    
    if SINGLE_CHUNK_PER_DOCUMENT:
        # Create single chunk per document
        logger.info("Creating single chunk per document")
        return [{
            "content": text,
            "metadata": metadata,
            "chunk_index": 0,
            "total_chunks": 1
        }]
    else:
        # Use traditional chunking
        chunks = chunk_text(text, chunk_size, overlap)
        return [{
            "content": chunk,
            "metadata": {**metadata, "chunk_index": i, "total_chunks": len(chunks)},
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i, chunk in enumerate(chunks)]
