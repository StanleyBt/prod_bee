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
    video_info = mapping.get(document_filename)
    
    if video_info:
        # Handle new format with segments
        if isinstance(video_info, dict) and "video_url" in video_info:
            video_url = video_info["video_url"]
        # Handle old format (simple string)
        elif isinstance(video_info, str):
            video_url = video_info
        else:
            video_url = None
            
        if video_url:
            logger.debug(f"Found video URL for {document_filename}: {video_url}")
        else:
            logger.debug(f"No video URL found for {document_filename}")
        
        return video_url
    else:
        logger.debug(f"No video URL found for {document_filename}")
        return None

def get_video_segment_for_document(tenant_id: str, document_filename: str, user_query: str) -> Optional[dict]:
    """
    Get video segment for a specific document based on user query.
    
    Args:
        tenant_id: The tenant ID
        document_filename: The filename of the document
        user_query: The user's query to match against segments
        
    Returns:
        Dictionary with video_url, start time, and topic if found, None otherwise
    """
    mapping = load_video_mapping(tenant_id)
    video_info = mapping.get(document_filename)
    
    if not video_info:
        return None
    
    # Handle old format (simple string)
    if isinstance(video_info, str):
        return {
            "url": video_info,
            "start": 0,
            "topic": "full_video"
        }
    
    # Handle new format with segments
    if isinstance(video_info, dict) and "video_url" in video_info:
        video_url = video_info["video_url"]
        segments = video_info.get("segments", {})
        
        if not segments:
            return {
                "url": video_url,
                "start": 0,
                "topic": "full_video"
            }
        
        # Find best matching segment
        best_segment = find_best_segment(user_query, segments)
        if best_segment:
            return {
                "url": video_url,
                "start": best_segment["start"],
                "topic": best_segment["topic"]
            }
        else:
            return {
                "url": video_url,
                "start": 0,
                "topic": "full_video"
            }
    
    return None

def find_best_segment(user_query: str, segments: dict) -> Optional[dict]:
    """
    Find the best matching segment based on user query keywords.
    
    Args:
        user_query: The user's query
        segments: Dictionary of segments with keywords
        
    Returns:
        Best matching segment with topic added, or None
    """
    query_lower = user_query.lower()
    best_match = None
    best_score = 0
    
    for topic, segment in segments.items():
        keywords = segment.get("keywords", [])
        score = calculate_keyword_match(query_lower, keywords)
        
        if score > best_score:
            best_score = score
            best_match = {
                "start": segment["start"],
                "topic": topic,
                "keywords": keywords
            }
    
    # Only return if we have a reasonable match (at least 1 keyword)
    return best_match if best_score > 0 else None

def calculate_keyword_match(query: str, keywords: list) -> int:
    """
    Calculate how many keywords from the segment match the user query.
    
    Args:
        query: The user's query (lowercase)
        keywords: List of keywords for the segment
        
    Returns:
        Number of matching keywords
    """
    if not keywords:
        return 0
    
    matches = 0
    for keyword in keywords:
        if keyword.lower() in query:
            matches += 1
    
    return matches

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
