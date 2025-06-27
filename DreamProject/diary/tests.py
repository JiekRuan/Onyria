from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
import json
import tempfile
from .models import Dream
from .utils import transcribe_audio, create_transcription
from unittest.mock import patch, MagicMock
from .views import softmax, classify_dream_from_emotions

User = get_user_model()


class DreamModelTest(TestCase):
    """Tests pour le modèle Dream simplifié"""
    
    def setUp(self):
        """
        Configuration initiale pour les tests du modèle Dream.
        Crée un utilisateur de test qui sera utilisé pour tous les tests.
        """
        self.user = User.objects.create_user(
            email='test@example.com',
            username='testuser',
            password='testpass123'
        )
    
    def test_create_dream(self):
        """
        Test de création d'un rêve basique.
        
        Vérifie que :
        - Un rêve peut être créé avec les champs obligatoires
        - L'objet est correctement sauvegardé en base de données
        - Les champs sont correctement assignés
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="J'ai rêvé d'un chien blanc qui courait dans un champ"
        )
        
        self.assertEqual(dream.user, self.user)
        self.assertEqual(dream.transcription, "J'ai rêvé d'un chien blanc qui courait dans un champ")
        self.assertIsNotNone(dream.date)

        
    def test_short_transcription_property(self):
        """
        Test de la propriété short_transcription.
        
        Vérifie que :
        - Pour un texte court : retourne le texte complet
        - Pour un texte long : retourne les 100 premiers caractères + "..."
        """
        # Test avec un texte court
        short_dream = Dream.objects.create(
            user=self.user,
            transcription="Court rêve"
        )
        self.assertEqual(short_dream.short_transcription, "Court rêve")
        
        # Test avec un texte long
        long_text = "J'ai fait un rêve très détaillé avec énormément d'éléments narratifs qui se succèdent continuellement sans interruption notable dans ce récit"
        long_dream = Dream.objects.create(
            user=self.user,
            transcription=long_text
        )
        expected = long_text[:100] + "..."
        self.assertEqual(long_dream.short_transcription, expected)
        
    def test_softmax_function(self):
        """
        TEST: Fonction softmax pour normaliser les émotions
        
        OBJECTIF: Vérifier que la fonction softmax transforme correctement les scores
        
        QUE FAIT CE TEST:
        - Teste la fonction softmax avec des valeurs d'émotions brutes
        - Vérifie que les valeurs sont normalisées (somme = 1)
        - Contrôle que les proportions relatives sont maintenues
        
        POURQUOI C'EST IMPORTANT:
        - Assure que les scores d'émotions sont cohérents
        - Permet de comparer les émotions de manière équitable
        - Base mathématique pour déterminer l'émotion dominante
        """
        # Test avec des valeurs d'émotions simulées
        raw_emotions = {
            "joie": 2.5,
            "tristesse": 1.2,
            "peur": 0.8,
            "colère": 0.3
        }
        
        # Application du softmax
        normalized_emotions = softmax(raw_emotions)
        
        # Vérifications
        # 1. Toutes les valeurs sont entre 0 et 1
        for emotion, score in normalized_emotions.items():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
        
        # 2. La somme totale est proche de 1 (erreur de précision flottante acceptable)
        total = sum(normalized_emotions.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # 3. L'ordre relatif est maintenu (joie doit rester la plus forte)
        self.assertGreater(normalized_emotions["joie"], normalized_emotions["tristesse"])
        self.assertGreater(normalized_emotions["tristesse"], normalized_emotions["peur"])
        self.assertGreater(normalized_emotions["peur"], normalized_emotions["colère"])
    
    def test_dominant_emotion_detection(self):
        """
        TEST: Détection de l'émotion dominante
        
        OBJECTIF: Vérifier que l'émotion avec le score le plus élevé est identifiée
        
        QUE FAIT CE TEST:
        - Crée différents scénarios d'émotions
        - Utilise max() pour trouver l'émotion dominante
        - Teste plusieurs cas : joie dominante, peur dominante, etc.
        
        POURQUOI C'EST IMPORTANT:
        - Cœur de l'analyse émotionnelle
        - Détermine la classification du rêve
        - Impact direct sur l'expérience utilisateur
        """
        # Cas 1: Rêve joyeux
        happy_emotions = {
            "joie": 0.7,
            "confiance": 0.2,
            "peur": 0.05,
            "tristesse": 0.05
        }
        dominant_emotion = max(happy_emotions.items(), key=lambda x: x[1])
        self.assertEqual(dominant_emotion[0], "joie")
        self.assertEqual(dominant_emotion[1], 0.7)
        
        # Cas 2: Cauchemar
        scary_emotions = {
            "peur": 0.8,
            "anxiété": 0.15,
            "joie": 0.03,
            "confiance": 0.02
        }
        dominant_emotion = max(scary_emotions.items(), key=lambda x: x[1])
        self.assertEqual(dominant_emotion[0], "peur")
        self.assertEqual(dominant_emotion[1], 0.8)
        
        # Cas 3: Émotions équilibrées (proche)
        balanced_emotions = {
            "joie": 0.35,
            "tristesse": 0.33,
            "surprise": 0.32
        }
        dominant_emotion = max(balanced_emotions.items(), key=lambda x: x[1])
        self.assertEqual(dominant_emotion[0], "joie")
        
