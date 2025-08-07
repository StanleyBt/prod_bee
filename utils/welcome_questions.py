#!/usr/bin/env python3
"""
Utility module for managing welcome questions from JSON files.
Supports frontend role-based filtering.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=10)  # Cache for performance
def load_welcome_questions(tenant_id: str) -> Dict:
    """
    Load welcome questions for a specific tenant from JSON file.
    
    Args:
        tenant_id: The tenant ID to load questions for
        
    Returns:
        Dictionary containing questions organized by module and role
    """
    try:
        # Construct path to welcome_questions.json
        questions_path = Path("data") / tenant_id / "welcome_questions.json"
        
        if not questions_path.exists():
            logger.warning(f"Welcome questions file not found for tenant {tenant_id}: {questions_path}")
            return {}
        
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
        logger.info(f"Loaded welcome questions for tenant {tenant_id}")
        return questions
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in welcome questions file for tenant {tenant_id}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading welcome questions for tenant {tenant_id}: {e}")
        return {}

def get_welcome_questions_by_role(tenant_id: str, role: str) -> Dict:
    """
    Get welcome questions filtered by role (for backend filtering if needed).
    
    Args:
        tenant_id: The tenant ID
        role: The user role (hr, employee, contractor)
        
    Returns:
        Dictionary with questions filtered by role
    """
    questions = load_welcome_questions(tenant_id)
    
    if not questions:
        return {}
    
    # HR role can see ALL questions from all roles
    if role.lower() == "hr":
        return questions
    
    # Other roles only see their own questions
    filtered_questions = {}
    for module, module_questions in questions.items():
        if role.lower() in module_questions:
            filtered_questions[module] = {role.lower(): module_questions[role.lower()]}
    
    return filtered_questions

def get_available_modules(tenant_id: str) -> List[str]:
    """
    Get list of available modules for a tenant.
    
    Args:
        tenant_id: The tenant ID
        
    Returns:
        List of available module names
    """
    questions = load_welcome_questions(tenant_id)
    return list(questions.keys()) if questions else []

def get_available_roles(tenant_id: str) -> List[str]:
    """
    Get list of available roles for a tenant.
    
    Args:
        tenant_id: The tenant ID
        
    Returns:
        List of available role names
    """
    questions = load_welcome_questions(tenant_id)
    
    if not questions:
        return []
    
    # Get all unique roles across all modules
    all_roles = set()
    for module_questions in questions.values():
        all_roles.update(module_questions.keys())
    
    return list(all_roles)

def get_random_sample_questions(
    tenant_id: str, 
    role: str = "hr",  # Default to hr to get all questions
    count: int = 3
) -> List[str]:
    """
    Get a random sample of questions for quick start.
    
    Args:
        tenant_id: The tenant ID
        role: The user role (defaults to hr to get all questions)
        count: Number of questions to return
        
    Returns:
        List of sample questions
    """
    import random
    
    # If role is hr, get all questions, otherwise filter by role
    if role.lower() == "hr":
        questions = load_welcome_questions(tenant_id)
    else:
        questions = get_welcome_questions_by_role(tenant_id, role)
    
    if not questions:
        return []
    
    # Flatten all questions into a single list
    all_questions = []
    for module_questions in questions.values():
        for role_questions in module_questions.values():
            all_questions.extend(role_questions)
    
    # Return random sample
    return random.sample(all_questions, min(count, len(all_questions)))

 