import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

from utils.pdf_processing import extract_chunks_from_pdf
from services.vector_db import store_chunks_batch

import warnings
warnings.filterwarnings("ignore", message="builtin type swigvarlink has no __module__ attribute")


HASH_FILE = ".ingest_hashes.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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

            pdf_files = list(module_dir.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDFs found in {module_dir}")
                continue

            for pdf_file in pdf_files:
                rel_path = f"{tenant_id}/{module}/{pdf_file.name}"
                current_hash = compute_file_hash(pdf_file)

                if existing_hashes.get(rel_path) == current_hash:
                    logger.info(f"Skipping unchanged file: {rel_path}")
                    continue

                logger.info(f"Ingesting file: {rel_path}")
                chunks = extract_chunks_from_pdf(str(pdf_file))
                chunk_objs = [{
                    "text": chunk,
                    "tenant_id": tenant_id,
                    "module": module,
                    "document_name": pdf_file.stem
                } for chunk in chunks]

                store_chunks_batch(chunk_objs)
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
