import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

from utils.pdf_processing import extract_chunks_from_pdf
from services.per_tenant_storage import store_document_chunks, initialize_per_tenant_storage
from utils.video_mapping import get_video_url_for_document  # NEW: Import video mapping utility

import warnings
warnings.filterwarnings("ignore", message="builtin type swigvarlink has no __module__ attribute")


HASH_FILE = ".ingest_hashes.json"

from core.logging_config import initialize_logging, get_logger

# Initialize logging
initialize_logging()
logger = get_logger(__name__)
 
def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()

def load_hashes() -> Dict[str, str]:
    """Load the JSON file containing previously computed file hashes."""
    if Path(HASH_FILE).exists():
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hashes: Dict[str, str]) -> None:
    """Save the updated file hashes to disk."""
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

def ingest_all_documents(base_path: str = "data") -> None:
    """
    Recursively ingest all PDF documents under the base_path.
    Skips files that have not changed based on SHA256 hash.
    """
    # Initialize per-tenant storage for ingestion
    if not initialize_per_tenant_storage():
        logger.error("Failed to initialize per-tenant storage for ingestion.")
        return

    base_dir = Path(base_path)
    if not base_dir.exists():
        logger.error(f"Data directory {base_path} does not exist.")
        return

    existing_hashes = load_hashes()
    updated_hashes = {}

    logger.info("Starting ingestion process...")

    any_files_ingested = False

    for tenant_dir in base_dir.iterdir():
        if not tenant_dir.is_dir():
            continue
        tenant_id = tenant_dir.name
        logger.info(f"Processing tenant: {tenant_id}")

        for module_dir in tenant_dir.iterdir():
            if not module_dir.is_dir():
                continue
            module = module_dir.name
            logger.info(f"Processing module: {module}")

            for role_dir in module_dir.iterdir():
                if not role_dir.is_dir():
                    continue
                role = role_dir.name
                logger.info(f"Processing role: {role}")

                pdf_files = list(role_dir.glob("*.pdf"))
                if not pdf_files:
                    logger.warning(f"No PDFs found in {role_dir}")
                    continue

                for pdf_file in pdf_files:
                    rel_path = f"{tenant_id}/{module}/{role}/{pdf_file.name}"
                    current_hash = compute_file_hash(pdf_file)

                    if existing_hashes.get(rel_path) == current_hash:
                        logger.info(f"Skipping unchanged file: {rel_path}")
                        continue

                    logger.info(f"Ingesting file: {rel_path}")
                    chunk_data = extract_chunks_from_pdf(str(pdf_file))
                    
                    # Get video URL for this document
                    video_url = get_video_url_for_document(tenant_id, pdf_file.name)
                    if video_url:
                        logger.info(f"Found video URL for {pdf_file.name}: {video_url}")
                    else:
                        logger.info(f"No video URL found for {pdf_file.name}")
                    
                    # Convert to the format expected by store_document_chunks
                    chunk_objs = []
                    for chunk_info in chunk_data:
                        # Merge document metadata with chunk metadata
                        enhanced_metadata = {
                            **chunk_info["metadata"],
                            "tenant_id": tenant_id,
                            "module": module,
                            "role": role,
                            "document_name": pdf_file.stem,
                            "chunk_index": chunk_info["chunk_index"],
                            "total_chunks": chunk_info["total_chunks"]
                        }
                        
                        chunk_objs.append({
                            "content": chunk_info["content"],
                            "metadata": enhanced_metadata
                        })

                    store_document_chunks(tenant_id, chunk_objs, module, role, video_url=video_url)
                    updated_hashes[rel_path] = current_hash
                    any_files_ingested = True

    # Merge updated hashes with existing and save
    existing_hashes.update(updated_hashes)
    save_hashes(existing_hashes)

    if any_files_ingested:
        logger.info("Ingestion complete: new or updated files processed.")
    else:
        logger.info("Ingestion complete: no new or updated files found.")

# if __name__ == "__main__":
#     logger.info("Running ingestion script as main program.")
#     ingest_all_documents()
