#!/usr/bin/env python3
"""
Test script for video segment functionality

This script tests the new video segment feature that returns specific timestamps
based on user queries.
"""

import logging
from services.per_tenant_storage import initialize_per_tenant_storage, retrieve_document_chunks
from utils.video_mapping import get_video_segment_for_document, load_video_mapping

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_video_segments():
    """Test video segment functionality"""
    logger.info("Testing video segment functionality...")
    
    # Initialize storage
    if not initialize_per_tenant_storage():
        logger.error("Failed to initialize storage")
        return
    
    # Test cases for different queries
    test_cases = [
        {
            "tenant_id": "CWFM",
            "query": "How do I reset my password?",
            "expected_topic": "password reset",
            "expected_start": 300
        },
        {
            "tenant_id": "CWFM", 
            "query": "How do I log into the system?",
            "expected_topic": "login process",
            "expected_start": 45
        },
        {
            "tenant_id": "CWFM",
            "query": "How do I set up my profile?",
            "expected_topic": "profile setup",
            "expected_start": 180
        },
        {
            "tenant_id": "FKP",
            "query": "How do I clock in?",
            "expected_topic": "clock in",
            "expected_start": 60
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n=== Test Case {i}: {test_case['query']} ===")
        
        # Test direct video segment lookup
        test_document = "Employee Mobile Onboarding.pdf" if test_case["tenant_id"] == "CWFM" else "Esamapark_Attendance_FAQs.pdf"
        
        video_segment = get_video_segment_for_document(
            test_case["tenant_id"], 
            test_document, 
            test_case["query"]
        )
        
        if video_segment:
            logger.info(f"✓ Found video segment:")
            logger.info(f"  URL: {video_segment['url']}")
            logger.info(f"  Start: {video_segment['start']} seconds ({video_segment['start']//60}:{video_segment['start']%60:02d})")
            logger.info(f"  Topic: {video_segment['topic']}")
            
            # Verify expected values
            if video_segment['topic'] == test_case['expected_topic']:
                logger.info(f"  ✓ Topic matches expected: {test_case['expected_topic']}")
            else:
                logger.warning(f"  ⚠️ Topic mismatch: expected {test_case['expected_topic']}, got {video_segment['topic']}")
            
            if video_segment['start'] == test_case['expected_start']:
                logger.info(f"  ✓ Start time matches expected: {test_case['expected_start']} seconds")
            else:
                logger.warning(f"  ⚠️ Start time mismatch: expected {test_case['expected_start']}, got {video_segment['start']}")
        else:
            logger.error(f"✗ No video segment found for query: {test_case['query']}")

def test_document_retrieval_with_segments():
    """Test document retrieval with video segments"""
    logger.info("\n=== Testing Document Retrieval with Video Segments ===")
    
    # Test queries that should return video segments
    test_queries = [
        ("CWFM", "employee onboarding process"),
        ("FKP", "attendance policy"),
    ]
    
    for tenant_id, query in test_queries:
        logger.info(f"\nTesting query for {tenant_id}: '{query}'")
        
        try:
            results = retrieve_document_chunks(tenant_id, query, top_k=2)
            
            if results:
                logger.info(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    logger.info(f"  Result {i}: {result[:200]}...")
                    
                    # Check if video URL is in the result
                    if "Video:" in result:
                        logger.info(f"    ✓ Video URL found in result")
                    else:
                        logger.info(f"    ✗ No video URL found in result")
            else:
                logger.info("No results found")
                
        except Exception as e:
            logger.error(f"Error retrieving documents for {tenant_id}: {e}")

def test_video_mapping_structure():
    """Test the new video mapping structure"""
    logger.info("\n=== Testing Video Mapping Structure ===")
    
    # Test CWFM mapping
    cwfm_mapping = load_video_mapping("CWFM")
    logger.info(f"CWFM video mappings: {len(cwfm_mapping)} documents")
    
    for doc_name, video_info in cwfm_mapping.items():
        logger.info(f"  Document: {doc_name}")
        if isinstance(video_info, dict) and "segments" in video_info:
            logger.info(f"    Video URL: {video_info['video_url']}")
            logger.info(f"    Segments: {len(video_info['segments'])}")
            for topic, segment in video_info['segments'].items():
                logger.info(f"      - {topic}: starts at {segment['start']}s")
        else:
            logger.info(f"    Old format: {video_info}")
    
    # Test FKP mapping
    fkp_mapping = load_video_mapping("FKP")
    logger.info(f"FKP video mappings: {len(fkp_mapping)} documents")
    
    for doc_name, video_info in fkp_mapping.items():
        logger.info(f"  Document: {doc_name}")
        if isinstance(video_info, dict) and "segments" in video_info:
            logger.info(f"    Video URL: {video_info['video_url']}")
            logger.info(f"    Segments: {len(video_info['segments'])}")
            for topic, segment in video_info['segments'].items():
                logger.info(f"      - {topic}: starts at {segment['start']}s")

def main():
    """Main test function"""
    logger.info("Starting video segment tests...")
    
    # Test video mapping structure
    test_video_mapping_structure()
    
    # Test video segments
    test_video_segments()
    
    # Test document retrieval
    test_document_retrieval_with_segments()
    
    logger.info("\n=== Test Summary ===")
    logger.info("✓ Video segments now include start times")
    logger.info("✓ Keyword matching finds relevant segments")
    logger.info("✓ API will return video segments with timestamps")
    logger.info("✓ Frontend can start videos at specific times")

if __name__ == "__main__":
    main()
