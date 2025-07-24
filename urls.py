"""
URL Configuration for Video_Txt project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import render

urlpatterns = [
    path('admin/', admin.site.urls),
    # Serve the main index page from frontend
    path('', lambda request: render(request, 'index.html')),
    # Include app URLs
    path('', include('app.urls')),
    path('analyzer/', include('analyzer.urls')),
]

# Add URL patterns for serving media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
