from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
import os

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('diary/', include('diary.urls')),
]

if settings.DEBUG:
    BASE_DIR = settings.BASE_DIR
    urlpatterns += static(settings.STATIC_URL, document_root=os.path.join(BASE_DIR, "diary", "static"))
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
