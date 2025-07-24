"""
Models for the Video_Txt Django application.
"""

from django.db import models
from django.conf import settings

class Upload(models.Model):
    """Model for storing information about uploaded files"""
    file = models.FileField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    file_size = models.FloatField()  # Size in MB
    upload_date = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    
    def __str__(self):
        return self.filename

class Transcript(models.Model):
    """Model for storing transcription results"""
    upload = models.OneToOneField(Upload, on_delete=models.CASCADE)
    raw_transcript = models.TextField()
    corrected_transcript = models.TextField(blank=True, null=True)
    language = models.CharField(max_length=50)
    duration_seconds = models.FloatField()
    duration_formatted = models.CharField(max_length=10)  # HH:MM:SS
    
    def __str__(self):
        return f"Transcript for {self.upload.filename}"

class Analysis(models.Model):
    """Model for storing analysis results"""
    transcript = models.OneToOneField(Transcript, on_delete=models.CASCADE)
    summary = models.TextField()
    keywords = models.TextField()
    sentiment_analysis = models.JSONField(default=dict)
    emotion_analysis = models.JSONField(default=dict)
    content_assessment = models.JSONField(default=dict)
    strengths_improvements = models.JSONField(default=dict)
    repeated_words = models.JSONField(default=dict)
    filler_words = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Analysis for {self.transcript.upload.filename}"
