from django.urls import path
from . import views

urlpatterns = [
    path('', views.dream_diary_view, name='dream_diary'),
    path('record/', views.dream_recorder_view, name='dream-recorder'),
    path('transcribe/', views.transcribe, name='transcribe'),
    path('analyse_from_voice/', views.analyse_from_voice, name='analyse_from_voice'),
]
