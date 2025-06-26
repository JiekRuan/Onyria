from django.urls import path
from .views import analyser_texte_view

urlpatterns = [
    path('', analyser_texte_view, name='diary'),
]
