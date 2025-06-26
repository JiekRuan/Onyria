from django.urls import path
from . import views
from .views import analyser_texte_view

urlpatterns = [
    path('', analyser_texte_view, name='diary'),
    path('record/', views.diary, name='dream-recorder'),
    path('transcribe/', views.transcribe, name='transcribe'),
]
