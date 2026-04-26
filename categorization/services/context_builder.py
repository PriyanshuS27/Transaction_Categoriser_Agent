"""
Context builder for transaction categorization.

Builds structured prompts from transaction data to be sent to the LLM.
"""

import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds a structured prompt context for the LLM based on transaction inputs.
    
    This class takes transaction data, historical examples, and account categories,
    then formats them into a well-structured prompt for the Groq API.
    """

    @staticmethod
    def build_prompt(
        description: str,
        vendor: str,
        industry: str,
        chart_of_accounts: List[str],
        historical_transactions: List[Dict[str, str]],
        amount: float = None,
    ) -> str:
        """
        Build a structured prompt for the LLM.
        
        Args:
            description: Transaction description
            vendor: Vendor/payee name
            industry: Company industry
            chart_of_accounts: Valid account categories
            historical_transactions: Previous transactions for context
            amount: Transaction amount (optional)
            
        Returns:
            A formatted prompt string for the LLM
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # ========================================================================
        # INPUT VALIDATION
        # ========================================================================
        # Validate all required parameters are present and have correct type
        if not description or not isinstance(description, str):
            raise ValueError("Transaction description is required and must be a string")
        vendor = vendor or "Unknown Vendor"
        if not chart_of_accounts or not isinstance(chart_of_accounts, list):
            raise ValueError("chart_of_accounts is required and must be a non-empty list")

        # ========================================================================
        # PROMPT BUILDING - Structured for Maximum LLM Accuracy
        # ========================================================================
        # The prompt structure is critical - organized information helps LLM
        # understand context and make better decisions
        
        # Part 1: Role Definition
        # Sets the LLM's context - "you are an expert accountant"
        prompt = "You are an expert accounting categorization assistant.\n\n"
        
        # Part 2: Clear Task Description
        # Tell the LLM exactly what to do - remove ambiguity
        prompt += "CATEGORIZATION TASK:\n"
        prompt += f"Categorize the following transaction into ONE of the provided account categories.\n\n"

        # ========================================================================
        # Part 3: Transaction Details (THE DATA TO CATEGORIZE)
        # ========================================================================
        prompt += "TRANSACTION DETAILS:\n"
        prompt += f"- Description: {description}\n"
        prompt += f"- Vendor: {vendor}\n"
        if amount is not None:
            # Include amount for better context (some expenses are category-dependent on size)
            prompt += f"- Amount: {amount}\n"
        prompt += f"- Industry: {industry}\n\n"

        # ========================================================================
        # Part 4: Valid Categories (CONSTRAINT - restrict to valid outputs)
        # ========================================================================
        # Enumerate categories so LLM knows exactly what to choose from
        # Prevents hallucination of invalid categories
        prompt += "VALID ACCOUNT CATEGORIES:\n"
        for i, category in enumerate(chart_of_accounts, 1):
            prompt += f"{i}. {category}\n"
        prompt += "\n"

        # ========================================================================
        # Part 5: Historical Examples (FEW-SHOT LEARNING)
        # ========================================================================
        # Provide similar examples so LLM understands categorization patterns
        # Few-shot learning significantly improves LLM accuracy
        # Examples are worth thousands of words in prompt engineering
        if historical_transactions:
            prompt += "HISTORICAL EXAMPLES (for reference):\n"
            for i, transaction in enumerate(historical_transactions, 1):
                trans_desc = transaction.get('description', 'N/A')
                trans_cat = transaction.get('category', 'N/A')
                # Format: "Description -> Category" for easy pattern matching
                prompt += f"{i}. Description: '{trans_desc}' -> Category: '{trans_cat}'\n"
            prompt += "\n"

        # ========================================================================
        # Part 6: Step-by-Step Instructions
        # ========================================================================
        # Chain-of-thought: ask LLM to show its reasoning
        # This improves accuracy and helps us understand the categorization
        prompt += "INSTRUCTIONS:\n"
        prompt += "1. Analyze the transaction and select the BEST matching category from the list above.\n"
        prompt += "2. Provide a confidence score (0.0 to 1.0) based on how certain you are.\n"
        prompt += "3. Explain your reasoning in 1-2 sentences.\n"
        prompt += "4. Suggest up to 2 alternative categories with lower confidence scores.\n"
        prompt += "5. Return ONLY a valid JSON response with NO markdown, NO code blocks, NO extra text.\n\n"

        # ========================================================================
        # Part 7: Expected Output Format (JSON Schema)
        # ========================================================================
        # Give exact JSON format so LLM knows how to structure response
        # ResponseParser will validate this format
        prompt += "REQUIRED JSON RESPONSE FORMAT (strict):\n"
        prompt += "{\n"
        prompt += '  "suggested_category": "category name from the list",\n'
        prompt += '  "confidence_score": 0.85,\n'
        prompt += '  "reasoning": "explanation of why this category was chosen",\n'
        prompt += '  "alternatives": [\n'
        prompt += '    {"category": "alternative 1", "confidence": 0.10},\n'
        prompt += '    {"category": "alternative 2", "confidence": 0.05}\n'
        prompt += "  ]\n"
        prompt += "}\n\n"

        prompt += "Now, categorize this transaction:"

        logger.info(f"Context built successfully (length: {len(description)} chars)")
        return prompt
