"""
WSGI config for Onyria project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
_raw = os.getenv("GROQ_API_KEY") or ""
os.environ["GROQ_API_KEY"] = _raw.replace("\r", "").replace("\n", "").strip()

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Onyria.settings')

application = get_wsgi_application()
