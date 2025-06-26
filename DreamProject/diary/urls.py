from django.urls import path
from . import views

urlpatterns = [
    path('transcription/', views.transcription_view, name="transcription"),
    path('emotions/', views.analyse_emotions_view, name="emotions"),
    path('prompt/', views.generate_prompt_view, name="prompt"),
    path('image/', views.generate_image_view, name="image"),
    path('interpretation/', views.interpret_dream_view, name="interpretation"),
]
