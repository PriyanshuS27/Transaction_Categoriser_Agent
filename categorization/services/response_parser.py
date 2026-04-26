"""
Response parser for LLM output.

Parses and validates LLM responses to ensure consistent JSON output.
"""

import json
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Parses and validates LLM responses for transaction categorization.
    
    Handles JSON extraction, validation, and fallback responses.
    """

    @staticmethod
    def parse_response(
        llm_response: str,
        chart_of_accounts: List[str]
    ) -> Dict[str, Any]:
        """
        Parse and validate LLM response.
        
        Args:
            llm_response: Raw response string from the LLM
            chart_of_accounts: Valid account categories for validation
            
        Returns:
            A validated categorization response dictionary
            
        Raises:
            ValueError: If response is empty or invalid format
        """
        if not llm_response or not isinstance(llm_response, str):
            logger.error("Invalid LLM response: empty or not a string")
            raise ValueError("LLM response must be a non-empty string")

        if not chart_of_accounts or not isinstance(chart_of_accounts, list):
            logger.error("Invalid chart_of_accounts")
            raise ValueError("chart_of_accounts must be a non-empty list")

        try:
            # Extract JSON from response
            # LLM might include markdown code blocks or extra text - parse carefully
            json_dict = ResponseParser._extract_json(llm_response)
            
            # Validate the JSON structure against our requirements
            # Ensures: required fields present, values in valid ranges, categories valid
            validated_response = ResponseParser._validate_and_normalize(
                json_dict,
                chart_of_accounts
            )
            
            logger.info(f"Successfully parsed response with category: {validated_response['suggested_category']}")
            return validated_response
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # If parsing/validation fails, use fallback response
            # This ensures we always return something useful instead of error
            logger.warning(f"Failed to parse LLM response: {str(e)}. Using fallback.")
            return ResponseParser._create_fallback_response(chart_of_accounts)

    @staticmethod
    def _extract_json(response_text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response text.
        
        The LLM might include markdown code blocks or extra text.
        This method extracts the JSON portion robustly.
        
        Why this matters: LLMs sometimes wrap JSON in markdown blocks:
        ```json
        {"category": "Office Supplies"}
        ```
        
        Args:
            response_text: Raw LLM response (might contain markdown/extra text)
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            json.JSONDecodeError: If no valid JSON found
        """
        # Remove markdown code block markers (```json or ```)
        # Handles both with and without language specification
        cleaned = re.sub(r'```json\n?', '', response_text)
        cleaned = re.sub(r'```\n?', '', cleaned)
        
        # Try to find JSON object pattern { ... }
        # Use DOTALL so . matches newlines too
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            try:
                # Try parsing the extracted JSON object
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse extracted JSON: {str(e)}")
                # Fallback: try parsing the entire cleaned response
                return json.loads(cleaned)
        else:
            # Try parsing the cleaned response directly
            return json.loads(cleaned)

    @staticmethod
    def _validate_and_normalize(
        json_dict: Dict[str, Any],
        chart_of_accounts: List[str]
    ) -> Dict[str, Any]:
        """
        Validate and normalize the parsed JSON response.
        
        Performs multi-layer validation:
        1. Check required fields are present
        2. Validate data types and ranges
        3. Check category is in valid accounts list
        4. Normalize/round values
        
        Args:
            json_dict: The parsed JSON dictionary from LLM
            chart_of_accounts: Valid account categories (whitelist)
            
        Returns:
            A normalized and validated response dictionary
            
        Raises:
            ValueError: If required fields missing or values invalid
        """
        # ========================================================================
        # STEP 1: Check Required Fields
        # ========================================================================
        # Ensure LLM response has all 4 required fields
        required_fields = ['suggested_category', 'confidence_score', 'reasoning', 'alternatives']
        for field in required_fields:
            if field not in json_dict:
                raise ValueError(f"Missing required field: {field}")

        # ========================================================================
        # STEP 2: Validate Suggested Category
        # ========================================================================
        suggested_category = json_dict.get('suggested_category', '').strip()
        
        # SECURITY: Whitelist check - ensure category is from our valid list
        # Prevents LLM from inventing categories not in chart of accounts
        if suggested_category not in chart_of_accounts:
            logger.warning(f"Suggested category '{suggested_category}' not in chart of accounts. Using fallback.")
            raise ValueError(f"Invalid category: {suggested_category}")

        # ========================================================================
        # STEP 3: Validate Confidence Score
        # ========================================================================
        try:
            # Convert to float and check range [0.0, 1.0]
            confidence = float(json_dict.get('confidence_score', 0.0))
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        except (ValueError, TypeError):
            raise ValueError("Invalid confidence_score format")

        # ========================================================================
        # STEP 4: Validate and Normalize Alternatives
        # ========================================================================
        alternatives = json_dict.get('alternatives', [])
        if not isinstance(alternatives, list):
            raise ValueError("alternatives must be a list")

        # Validate each alternative separately
        # Skip invalid ones instead of failing completely
        validated_alternatives = []
        skipped_alternatives = 0
        for alt in alternatives:
            if not isinstance(alt, dict):
                skipped_alternatives += 1
                continue
            
            alt_category = alt.get('category', '').strip()
            alt_confidence = alt.get('confidence', 0.0)
            
            try:
                alt_confidence = float(alt_confidence)
                # Check: confidence in range AND category in whitelist
                if 0.0 <= alt_confidence <= 1.0 and alt_category in chart_of_accounts:
                    validated_alternatives.append({
                        'category': alt_category,
                        'confidence': round(alt_confidence, 2)  # Round to 2 decimals
                    })
                else:
                    skipped_alternatives += 1
                    logger.warning(f"Skipped invalid alternative: category='{alt_category}', confidence={alt_confidence}")
            except (ValueError, TypeError):
                # Invalid confidence - skip this alternative
                skipped_alternatives += 1
                logger.warning(f"Skipped alternative with invalid confidence value")
                continue
        
        if skipped_alternatives > 0:
            logger.warning(f"Total alternatives skipped: {skipped_alternatives}")

        # ========================================================================
        # STEP 5: Return Normalized Response
        # ========================================================================
        # Normalize all values for consistency
        return {
            'suggested_category': suggested_category,
            'confidence_score': round(confidence, 2),  # Round to 2 decimal places
            'reasoning': str(json_dict.get('reasoning', 'No reasoning provided'))[:1000],  # Max 1000 chars
            'alternatives': validated_alternatives[:2],  # Limit to 2 alternatives
            'status': 'success'
        }

    @staticmethod
    def _create_fallback_response(chart_of_accounts: List[str]) -> Dict[str, Any]:
        """
        Create a fallback response when parsing fails.
        
        Why fallback response:
        - Instead of crashing or returning error, we return a valid response
        - Uses first category with 0.0 confidence (indicates "best guess")
        - Better UX: user gets a result instead of error
        - Logging captures what went wrong for debugging
        
        Args:
            chart_of_accounts: Valid account categories (use first as default)
            
        Returns:
            A valid fallback response with low confidence (0.0)
        """
        if not chart_of_accounts:
            # This should not happen, but handle defensive programming
            chart_of_accounts = ['Other']

        logger.info("Using fallback response due to parsing error")
        
        # Return default response using first valid category
        # confidence_score = 0.0 signals that this is a fallback, not LLM-generated
        return {
            'suggested_category': chart_of_accounts[0],  # Select first category as default
            'confidence_score': 0.0,  # 0.0 confidence = fallback/default value
            'reasoning': 'Unable to parse LLM response. Using fallback categorization.',
            'alternatives': [],
            'status': 'fallback'
        }
