"""
URL configuration for transaction_categorizer project.
"""

from django.urls import path, include
from django.shortcuts import render

def home(request):
    """Serve the transaction categorizer UI."""
    return render(request, 'index.html')

urlpatterns = [
    path('', home, name='home'),
    path('api/', include('categorization.urls')),
]
