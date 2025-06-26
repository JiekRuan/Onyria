from django.urls import path
from . import views

urlpatterns = [
    path('record/', views.diary, name='dream-recorder'),
    path('transcribe/', views.transcribe, name='transcribe'),
]