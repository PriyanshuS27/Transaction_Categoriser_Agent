#!/usr/bin/env python
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.test import Client

client = Client()
response = client.get('/', HTTP_HOST='localhost')
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("✓ SUCCESS - Home page returned 200")
    print(f"Content length: {len(response.content)}")
else:
    print(f"✗ FAILED - Got {response.status_code}")
    content = response.content.decode('utf-8')[:500]
    print("Response preview:", content)
