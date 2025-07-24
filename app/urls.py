"""URL Configuration for the app module."""

from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from app import views

urlpatterns = [
    path('upload-video/', csrf_exempt(views.upload_video), name='upload_video'),
    path('pose-voice-analysis/', csrf_exempt(views.pose_voice_analysis), name='pose_voice_analysis'),
    path('generate-coach-feedback/', csrf_exempt(views.generate_coach_feedback), name='generate_coach_feedback'),
    path('health/', views.health_check, name='health_check'),
]
