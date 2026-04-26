"""
Serializers for transaction categorization API.
"""

from rest_framework import serializers


class HistoricalTransactionSerializer(serializers.Serializer):
    """
    Serializer for historical transaction examples.
    
    Attributes:
        description: Transaction description
        category: Assigned category
    """
    description = serializers.CharField(max_length=500)
    category = serializers.CharField(max_length=100, min_length=2)


class TransactionCategorizationSerializer(serializers.Serializer):
    """
    Main serializer for transaction categorization requests.
    
    Validates all required and optional fields for the categorization endpoint.
    """
    description = serializers.CharField(
        max_length=1000,
        help_text="Description of the transaction"
    )
    vendor = serializers.CharField(
        max_length=200,
        help_text="Vendor or payee name"
    )
    amount = serializers.DecimalField(
        max_digits=15,
        decimal_places=2,
        required=False,
        allow_null=True,
        min_value=0,
        help_text="Transaction amount"
    )
    company_id = serializers.CharField(
        max_length=100,
        help_text="Company identifier"
    )
    industry = serializers.CharField(
        max_length=100,
        help_text="Company industry"
    )
    chart_of_accounts = serializers.ListField(
        child=serializers.CharField(max_length=100),
        help_text="List of valid account categories"
    )
    historical_transactions = HistoricalTransactionSerializer(
        many=True,
        required=False,
        help_text="Previous transactions for context"
    )

    def validate_chart_of_accounts(self, value):
        """
        Validate that chart_of_accounts is not empty.
        
        Args:
            value: List of account categories
            
        Returns:
            The validated list
            
        Raises:
            ValidationError: If list is empty
        """
        if not value or len(value) == 0:
            raise serializers.ValidationError("chart_of_accounts must not be empty")
        return value

    def validate(self, data):
        """
        Cross-field validation.
        
        Args:
            data: All serializer data
            
        Returns:
            The validated data
        """
        if 'historical_transactions' not in data:
            data['historical_transactions'] = []
        return data


class AlternativeSerializer(serializers.Serializer):
    """
    Serializer for alternative category suggestions.
    
    Attributes:
        category: Alternative category name
        confidence: Confidence score (0-1)
    """
    category = serializers.CharField(max_length=100)
    confidence = serializers.FloatField(min_value=0.0, max_value=1.0)


class CategorizationResponseSerializer(serializers.Serializer):
    """
    Response serializer for categorization results.
    
    Attributes:
        suggested_category: The recommended category
        confidence_score: Confidence level (0-1)
        reasoning: Explanation for the suggestion
        alternatives: List of alternative categories
        status: Success/error status
    """
    suggested_category = serializers.CharField(max_length=100)
    confidence_score = serializers.FloatField(min_value=0.0, max_value=1.0)
    reasoning = serializers.CharField(max_length=1000)
    alternatives = AlternativeSerializer(many=True)
    status = serializers.CharField(max_length=20)
