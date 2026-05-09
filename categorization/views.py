"""
Views for transaction categorization API.
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.shortcuts import render

from categorization.serializers import (
    TransactionCategorizationSerializer,
    CategorizationResponseSerializer,
)
from categorization.services.context_builder import ContextBuilder
from categorization.services.llm_wrapper import LLMWrapper, GroqProvider
from categorization.services.response_parser import ResponseParser

logger = logging.getLogger(__name__)


def home(request):
    """Serve the transaction categorizer UI."""
    return render(request, 'index.html')


class CategorizationView(APIView):
    """
    API view for transaction categorization.
    
    Endpoint: POST /api/categorize/
    
    Accepts transaction details and returns suggested accounting category
    using LLM-powered analysis.
    """

    def post(self, request):
        """
        Categorize a transaction.
        
        Args:
            request: HTTP request with transaction data
            
        Returns:
            JSON response with categorization result
        """
        # ========================================================================
        # STEP 1: Validate Input Data
        # ========================================================================
        # Use DRF serializer to validate all required fields are present and valid
        serializer = TransactionCategorizationSerializer(data=request.data)
        
        if not serializer.is_valid():
            logger.warning(f"Invalid request data: {serializer.errors}")
            return Response(
                {
                    'status': 'error',
                    'message': 'Invalid request data',
                    'errors': serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Extract and use validated data (cleaner, safer than raw request.data)
            data = serializer.validated_data
            
            logger.info(
                f"Processing categorization request for: {data.get('description', '')[:50]}"
            )

            # ====================================================================
            # STEP 2: Build LLM Prompt with Context
            # ====================================================================
            # ContextBuilder creates a structured prompt with:
            # - Transaction details (description, vendor, amount)
            # - Valid account categories
            # - Historical examples (few-shot learning)
            # This improves LLM accuracy significantly
            prompt = ContextBuilder.build_prompt(
                description=data['description'],
                vendor=data['vendor'],
                industry=data['industry'],
                chart_of_accounts=data['chart_of_accounts'],
                historical_transactions=data.get('historical_transactions', []),
                amount=data.get('amount')
            )

            # ====================================================================
            # STEP 3: Initialize LLM Provider and Call API
            # ====================================================================
            # Retrieve API key from settings (loaded from .env)
            groq_api_key = settings.GROQ_API_KEY
            
            if not groq_api_key:
                logger.error("GROQ_API_KEY not configured")
                return Response(
                    {
                        'status': 'error',
                        'message': 'LLM service is not configured'
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            try:
                # Initialize Groq provider with API key
                provider = GroqProvider(api_key=groq_api_key)
                # Wrap provider in high-level interface
                llm_wrapper = LLMWrapper(provider)
                # Call LLM with prompt (includes automatic retry logic)
                llm_response = llm_wrapper.categorize(prompt)
            except ValueError as e:
                # Configuration errors (invalid API key, etc.)
                logger.error(f"LLM configuration error: {str(e)}")
                return Response(
                    {
                        'status': 'error',
                        'message': 'LLM service is not properly configured'
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            except Exception as e:
                # API call errors (network, timeout, API errors)
                # LLM wrapper already retries 3 times before raising
                logger.error(f"LLM API error: {str(e)}")
                return Response(
                    {
                        'status': 'error',
                        'message': 'Failed to get categorization from LLM service'
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            # ====================================================================
            # STEP 4: Parse and Validate LLM Response
            # ====================================================================
            # ResponseParser:
            # - Extracts JSON from LLM response (handles markdown code blocks)
            # - Validates required fields (suggested_category, confidence, reasoning)
            # - Checks category is in valid chart of accounts
            # - Returns fallback if parsing fails
            try:
                categorization_result = ResponseParser.parse_response(
                    llm_response=llm_response,
                    chart_of_accounts=data['chart_of_accounts']
                )
            except ValueError as e:
                logger.error(f"Response parsing error: {str(e)}")
                return Response(
                    {
                        'status': 'error',
                        'message': 'Failed to parse LLM response'
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            # ====================================================================
            # STEP 5: Validate Response Against Schema
            # ====================================================================
            # Double-check response format matches expected schema
            response_serializer = CategorizationResponseSerializer(
                data=categorization_result
            )
            
            if not response_serializer.is_valid():
                logger.error(
                    f"Invalid response format: {response_serializer.errors}"
                )
                return Response(
                    {
                        'status': 'error',
                        'message': 'Invalid response format from categorization service'
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # ====================================================================
            # STEP 6: Return Success Response
            # ====================================================================
            logger.info(
                f"Successfully categorized transaction as: {categorization_result['suggested_category']}"
            )
            
            return Response(
                categorization_result,
                status=status.HTTP_200_OK
            )

        except Exception as e:
            # Catch any unexpected errors not handled above
            logger.error(f"Unexpected error in categorization: {str(e)}", exc_info=True)
            return Response(
                {
                    'status': 'error',
                    'message': 'An unexpected error occurred during categorization'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
