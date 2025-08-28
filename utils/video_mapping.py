"""
Video Mapping Utility

This module handles the mapping between document filenames and their associated video URLs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def load_video_mapping(tenant_id: str) -> Dict[str, str]:
    """
    Load video mapping for a specific tenant.
    
    Args:
        tenant_id: The tenant ID (e.g., 'CWFM', 'FKP')
        
    Returns:
        Dictionary mapping document filenames to video URLs
    """
    mapping_file = Path(f"data/{tenant_id}/video_mapping.json")
    
    if not mapping_file.exists():
        logger.info(f"No video mapping file found for tenant {tenant_id}")
        return {}
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        logger.info(f"Loaded video mapping for tenant {tenant_id}: {len(mapping)} mappings")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load video mapping for tenant {tenant_id}: {e}")
        return {}

def get_video_url_for_document(tenant_id: str, document_filename: str) -> Optional[str]:
    """
    Get video URL for a specific document.
    
    Args:
        tenant_id: The tenant ID
        document_filename: The filename of the document (e.g., 'employee_onboarding.pdf')
        
    Returns:
        Video URL if found, None otherwise
    """
    mapping = load_video_mapping(tenant_id)
    video_url = mapping.get(document_filename)
    
    if video_url:
        logger.debug(f"Found video URL for {document_filename}: {video_url}")
    else:
        logger.debug(f"No video URL found for {document_filename}")
    
    return video_url

def save_video_mapping(tenant_id: str, mapping: Dict[str, str]) -> bool:
    """
    Save video mapping for a tenant.
    
    Args:
        tenant_id: The tenant ID
        mapping: Dictionary mapping document filenames to video URLs
        
    Returns:
        True if successful, False otherwise
    """
    mapping_file = Path(f"data/{tenant_id}/video_mapping.json")
    
    try:
        # Ensure directory exists
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Saved video mapping for tenant {tenant_id}: {len(mapping)} mappings")
        return True
    except Exception as e:
        logger.error(f"Failed to save video mapping for tenant {tenant_id}: {e}")
        return False

def add_video_mapping(tenant_id: str, document_filename: str, video_url: str) -> bool:
    """
    Add a single video mapping for a document.
    
    Args:
        tenant_id: The tenant ID
        document_filename: The filename of the document
        video_url: The video URL
        
    Returns:
        True if successful, False otherwise
    """
    mapping = load_video_mapping(tenant_id)
    mapping[document_filename] = video_url
    return save_video_mapping(tenant_id, mapping)

def remove_video_mapping(tenant_id: str, document_filename: str) -> bool:
    """
    Remove a video mapping for a document.
    
    Args:
        tenant_id: The tenant ID
        document_filename: The filename of the document
        
    Returns:
        True if successful, False otherwise
    """
    mapping = load_video_mapping(tenant_id)
    if document_filename in mapping:
        del mapping[document_filename]
        return save_video_mapping(tenant_id, mapping)
    return True
