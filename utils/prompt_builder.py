"""
Prompt Builder Module

Provides modular components for building RAG prompts.
Separates concerns for better maintainability and testing.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MemoryFormatter:
    """Handles formatting of conversation memories."""
    
    @staticmethod
    def format_memories(memories: List[Dict[str, Any]], max_memories: int = 3) -> str:
        """
        Format conversation memories into a readable string.
        
        Args:
            memories: List of memory dictionaries
            max_memories: Maximum number of memories to include
            
        Returns:
            Formatted memory string
        """
        if not memories:
            return ""
        
        memory_entries = []
        # Use only the most recent memories to keep context focused
        recent_memories = memories[:max_memories] if len(memories) > max_memories else memories
        
        for memory in recent_memories:
            if isinstance(memory, dict) and 'user' in memory and 'bot' in memory:
                user_msg = str(memory.get('user', ''))
                bot_msg = str(memory.get('bot', ''))
                memory_entries.append(f"Previous conversation:\nUser: {user_msg}\nAssistant: {bot_msg}")
        
        return "\n".join(memory_entries)


class ContextProcessor:
    """Handles processing and validation of document context."""
    
    @staticmethod
    def process_context(context: str) -> str:
        """
        Process and validate document context.
        
        Args:
            context: Raw context from document retrieval
            
        Returns:
            Processed context string
        """
        if not context or context.strip().lower() in ["no context found", "retrieval error"]:
            return "No specific documentation found for this query in the current module. The information you're looking for might be available in a different module."
        return context


class PromptTemplate:
    """Contains the main prompt template and instructions."""
    
    SYSTEM_PROMPT = (
        "You are a smart, friendly, and professional AI assistant built for a modern SaaS platform. "
        "Your goal is to help users with their questions based ONLY on the provided documentation.\n\n"
        "Instructions:\n"
        "- CRITICAL: Base your response STRICTLY on the module context provided below.\n"
        "- If the context is insufficient or unclear, say so rather than making assumptions.\n"
        "- Understand the user's intent using context and conversation history.\n"
        "- Respond clearly and concisely using markdown formatting — include bullet points, numbered steps, or headings if useful.\n"
        "- Avoid overwhelming users — provide just enough detail to address their current need.\n"
        "- Adapt your tone to be professional yet approachable.\n"
        "- Respect the user's role (e.g., employee, manager, vendor, HR) to customize instructions.\n"
        "- Keep responses relevant — don't repeat earlier answers unless needed for clarity.\n"
        "- Your responses should be modular and context-aware.\n"
        "- IMPORTANT: If the context doesn't contain relevant information, acknowledge that you don't have information about that specific topic in the current module.\n"
        "- When the current module doesn't have relevant information, suggest checking other related modules that might contain the information.\n"
        "- For attendance-related queries in Onboarding module, suggest checking the Attendance module.\n"
        "- For onboarding-related queries in Attendance module, suggest checking the Onboarding module.\n"
        "- Be professional and helpful: acknowledge the limitation and guide users to the right module.\n"
        "- End with a brief, encouraging note like 'Is there something else I can help you with?'\n"
        "- IMPORTANT: You have access to conversation history below. Use it to provide context-aware responses.\n"
        "- Be conversational and engaging, but stay focused on the user's needs.\n"
        "- IMPORTANT: Respond naturally to user inputs, understanding context from conversation history.\n"
        "- IMPORTANT: Adapt your response style based on the user's preferences below.\n"
        "- When you don't have specific information, ask clarifying questions to help find relevant content within the available documentation.\n\n"
    )
    
    CONTEXT_SECTION = "MODULE CONTEXT:\n{context}\n"
    
    CONVERSATION_HISTORY_SECTION = "Conversation History:\n{memory_str}\n\n"
    
    USER_INPUT_SECTION = (
        "User Input:\n{user_input}\n\n"
        "REMEMBER: Base your response ONLY on the context provided above. "
        "If the context doesn't contain enough information, acknowledge that you don't have information about that topic in the current module and suggest checking related modules. "
        "End with a brief, encouraging note to keep the conversation going.\n\n"
    )


class ConversationFlowHandler:
    """Handles conversation flow and ending instructions."""
    
    @staticmethod
    def get_ending_instruction(has_conversation_history: bool) -> str:
        """
        Get appropriate ending instruction based on conversation state.
        
        Args:
            has_conversation_history: Whether there is previous conversation
            
        Returns:
            Ending instruction string
        """
        if has_conversation_history:
            return "Continue the conversation naturally, building on our previous discussion and providing helpful, context-aware responses based on the available documentation."
        else:
            return "Engage with the user's input in a helpful and professional manner, focusing on the available documentation."


class PromptBuilder:
    """Main prompt builder that orchestrates all components."""
    
    def __init__(self):
        self.memory_formatter = MemoryFormatter()
        self.context_processor = ContextProcessor()
        self.template = PromptTemplate()
        self.flow_handler = ConversationFlowHandler()
    
    def build_prompt(self, state: Dict[str, Any], user_input: str) -> str:
        """
        Build a complete RAG prompt from the given state and user input.
        
        Args:
            state: RAG state containing memories, context, role, etc.
            user_input: The user's current input
            
        Returns:
            Complete formatted prompt
        """
        # Extract components from state
        memories = state.get("memories", [])
        user_preferences = state.get("user_preferences", "")
        context = state.get('context', '')
        role = state.get('role', 'unknown')
        
        # Debug logging
        logger.info(f"Building prompt with {len(memories)} memories and user preferences: {user_preferences[:100] if user_preferences else 'None'}")
        if memories:
            logger.info(f"Memory structure: {memories[0] if memories else 'No memories'}")
        
        # Process components
        memory_str = self.memory_formatter.format_memories(memories)
        processed_context = self.context_processor.process_context(context)
        
        # Log conversation history
        if memory_str:
            logger.info(f"Conversation history preview: {memory_str[:200]}...")
        else:
            logger.info("No conversation history found")
        
        # Determine conversation flow
        has_conversation_history = bool(memory_str.strip())
        ending_instruction = self.flow_handler.get_ending_instruction(has_conversation_history)
        
        # Build the complete prompt
        prompt = self._assemble_prompt(
            role=role,
            user_preferences=user_preferences,
            context=processed_context,
            memory_str=memory_str,
            user_input=user_input,
            ending_instruction=ending_instruction
        )
        
        return prompt
    
    def _assemble_prompt(self, role: str, user_preferences: str, context: str, 
                        memory_str: str, user_input: str, ending_instruction: str) -> str:
        """Assemble the final prompt from all components."""
        
        # Start with system prompt
        prompt = self.template.SYSTEM_PROMPT
        
        # Add role and preferences
        prompt += f"User Role (if known): {role}\n"
        prompt += f"User Preferences: {user_preferences if user_preferences else 'No specific preferences found'}\n"
        
        # Add context section
        prompt += self.template.CONTEXT_SECTION.format(context=context)
        
        # Add conversation history if available
        if memory_str:
            prompt += self.template.CONVERSATION_HISTORY_SECTION.format(memory_str=memory_str)
        
        # Add user input and ending
        prompt += self.template.USER_INPUT_SECTION.format(user_input=user_input)
        prompt += ending_instruction
        
        return prompt


# Convenience function for backward compatibility
def build_prompt(state: Dict[str, Any], user_input: str) -> str:
    """
    Build a complete RAG prompt from the given state and user input.
    
    This is a convenience function that maintains the same interface
    as the original build_prompt function.
    
    Args:
        state: RAG state containing memories, context, role, etc.
        user_input: The user's current input
        
    Returns:
        Complete formatted prompt
    """
    builder = PromptBuilder()
    return builder.build_prompt(state, user_input)
