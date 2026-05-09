"""
URL configuration for transaction_categorizer project.
"""

from django.urls import path, include
from categorization.views import home

urlpatterns = [
    path('', home, name='home'),
    path('api/', include('categorization.urls')),
]
