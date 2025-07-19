from django.urls import path
from . import views

urlpatterns = [
    path('', views.placeholder, name='diary'),
    path('record/', views.diary, name='dream-recorder'), 
    path('transcribe/', views.transcribe, name='transcribe'),
    path('analyse_from_voice/', views.analyse_from_voice, name='analyse_from_voice'),
]
