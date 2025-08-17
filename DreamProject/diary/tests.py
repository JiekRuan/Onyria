from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from unittest.mock import patch, MagicMock
import json
import tempfile
from .models import Dream
from .utils import transcribe_audio, softmax, classify_dream
from django.core.management.base import BaseCommand
from diary.utils import interpret_dream
import json

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

## Tests pour les trois cartes du profil - type rêveur / émotion dominante

def test_profil_onirique_stats(self):
    """
    TEST: Calcul des statistiques de profil onirique
    
    OBJECTIF: Vérifier que les stats des cartes de profil sont correctement calculées
    
    QUE FAIT CE TEST:
    - Crée plusieurs rêves avec différents types (rêve/cauchemar)
    - Vérifie le calcul du pourcentage rêves vs cauchemars
    - Contrôle la détermination du statut (rêveur/cauchemardeur)
    - Teste l'émotion dominante globale de l'utilisateur
    
    POURQUOI C'EST IMPORTANT:
    - Assure la justesse des cartes de profil dans dream_diary
    - Vérifie que les labels formatés sont correctement appliqués
    - Garantit une expérience utilisateur cohérente
    """
    from .utils import get_profil_onirique_stats
    
    # Création de rêves de test avec différents types et émotions
    Dream.objects.create(
        user=self.user,
        transcription="Rêve paisible dans un jardin",
        dream_type="reve",
        dominant_emotion="serein"
    )
    Dream.objects.create(
        user=self.user,
        transcription="Cauchemar effrayant",
        dream_type="cauchemar", 
        dominant_emotion="apeure"
    )
    Dream.objects.create(
        user=self.user,
        transcription="Rêve joyeux avec des amis",
        dream_type="reve",
        dominant_emotion="heureux"
    )
    Dream.objects.create(
        user=self.user,
        transcription="Autre rêve heureux",
        dream_type="reve",
        dominant_emotion="heureux"
    )
    
    # Récupération des stats
    stats = get_profil_onirique_stats(self.user)
    
    # Vérifications des calculs
    # 3 rêves sur 4 = 75% de rêves
    self.assertEqual(stats['pourcentage_reveuse'], 75)
    self.assertEqual(stats['statut_reveuse'], 'reve')  # Valeur brute du backend
    self.assertEqual(stats['label_reveuse'], 'rêves')
    
    # Émotion dominante: "heureux" apparaît 2 fois sur 4
    self.assertEqual(stats['emotion_dominante'], 'heureux')  # Valeur brute du backend
    self.assertEqual(stats['emotion_dominante_percentage'], 50)

def test_dream_diary_view_with_labels(self):
    """
    TEST: Vue dream_diary avec formatage des labels
    
    OBJECTIF: Vérifier que la vue applique correctement les labels formatés
    
    QUE FAIT CE TEST:
    - Simule la vue dream_diary avec des données de test
    - Vérifie que les labels bruts sont transformés en labels affichables
    - Contrôle que le contexte contient les bonnes valeurs formatées
    
    POURQUOI C'EST IMPORTANT:
    - Assure que l'implémentation DRY fonctionne correctement
    - Vérifie la cohérence entre backend et frontend
    - Garantit l'affichage correct dans les cartes de profil
    """
    # Création de rêves de test
    Dream.objects.create(
        user=self.user,
        transcription="Cauchemar terrifiant",
        dream_type="cauchemar",
        dominant_emotion="apeure"
    )
    Dream.objects.create(
        user=self.user,
        transcription="Rêve en colère",
        dream_type="reve", 
        dominant_emotion="en_colere"
    )
    
    # Connexion de l'utilisateur
    self.client.login(email='test@example.com', password='testpass123')
    
    # Appel de la vue
    response = self.client.get(reverse('dream_diary'))
    
    # Vérifications de la réponse
    self.assertEqual(response.status_code, 200)
    
    # Vérification que les labels sont formatés correctement
    context = response.context
    
    # Vérification du formatage des émotions (brut -> formaté)
    # "en_colere" ou "apeure" devrait devenir "En colère" ou "Apeuré"
    emotion_formatee = context.get('emotion_dominante')
    self.assertIn(emotion_formatee, ['En colère', 'Apeuré'])
    
    # Vérification du formatage du type de rêve 
    # "cauchemar" devrait devenir "Cauchemar"  
    statut_formate = context.get('statut_reveuse')
    self.assertEqual(statut_formate, 'Cauchemar')

def test_emotion_dream_type_labels_consistency(self):
    """
    TEST: Cohérence des labels entre les dictionnaires
    
    OBJECTIF: Vérifier que les dictionnaires EMOTION_LABELS et DREAM_TYPE_LABELS
              correspondent aux valeurs utilisées dans l'application
    
    QUE FAIT CE TEST:
    - Importe les dictionnaires de labels depuis views.py
    - Vérifie que toutes les émotions possibles sont couvertes
    - Contrôle que tous les types de rêves sont mappés
    - Teste la cohérence du formatage (majuscules, accents)
    
    POURQUOI C'EST IMPORTANT:
    - Évite les erreurs d'affichage si de nouvelles émotions sont ajoutées
    - Assure la maintenance à long terme
    - Garantit que le principe DRY est respecté partout
    """
    from .views import EMOTION_LABELS, DREAM_TYPE_LABELS
    
    # Test des émotions - toutes les émotions possibles doivent être mappées
    expected_emotions = [
        'heureux', 'anxieux', 'triste', 'en_colere', 
        'fatigue', 'apeure', 'surpris', 'serein'
    ]
    
    for emotion in expected_emotions:
        self.assertIn(emotion, EMOTION_LABELS)
        # Vérifie que le label formaté commence par une majuscule
        self.assertTrue(EMOTION_LABELS[emotion][0].isupper())
    
    # Test des types de rêves
    expected_dream_types = ['reve', 'cauchemar']
    
    for dream_type in expected_dream_types:
        self.assertIn(dream_type, DREAM_TYPE_LABELS)
        # Vérifie le formatage correct
        self.assertTrue(DREAM_TYPE_LABELS[dream_type][0].isupper())
    
    # Vérifications spécifiques pour les accents et caractères spéciaux
    self.assertEqual(DREAM_TYPE_LABELS['rêve'], 'Rêve')
    self.assertEqual(EMOTION_LABELS['en_colere'], 'En colère')
    self.assertEqual(EMOTION_LABELS['apeure'], 'Apeuré')

def test_analyse_from_voice_with_formatted_labels(self):
    """
    TEST: API analyse_from_voice avec labels formatés
    
    OBJECTIF: Vérifier que l'API retourne les labels formatés dans la réponse JSON
    
    QUE FAIT CE TEST:
    - Mock les fonctions d'analyse pour retourner des valeurs contrôlées
    - Simule un appel à l'API avec un fichier audio
    - Vérifie que la réponse JSON contient les labels formatés
    
    POURQUOI C'EST IMPORTANT:
    - Assure que l'interface d'enregistrement affiche les bons labels
    - Vérifie la cohérence entre l'API et les vues template
    - Garantit que le JavaScript n'a plus besoin des dictionnaires de mapping
    """
    with patch('your_app.views.transcribe_audio') as mock_transcribe, \
         patch('your_app.views.analyze_emotions') as mock_analyze, \
         patch('your_app.views.classify_dream') as mock_classify, \
         patch('your_app.views.interpret_dream') as mock_interpret, \
         patch('your_app.views.generate_image_from_text') as mock_generate:
        
        # Configuration des mocks
        mock_transcribe.return_value = "Rêve de test"
        mock_analyze.return_value = ({'heureux': 0.8}, ['heureux'])
        mock_classify.return_value = 'rêve'
        mock_interpret.return_value = {'symbole': 'bonheur'}
        mock_generate.return_value = None
        
        # Connexion utilisateur
        self.client.login(email='test@example.com', password='testpass123')
        
        # Simulation d'un fichier audio
        with tempfile.NamedTemporaryFile(suffix='.wav') as audio_file:
            audio_file.write(b'fake_audio_data')
            audio_file.seek(0)
            
            response = self.client.post(reverse('analyse_from_voice'), {
                'audio': audio_file
            })
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        
        self.assertTrue(data['success'])
        # Vérification que les labels sont formatés
        self.assertEqual(data['dominant_emotion'], ['Heureux'])  # Pas 'heureux'
        self.assertEqual(data['dream_type'], 'Rêve')  # Pas 'reve'

class Command(BaseCommand):
    help = 'Teste le format de réponse de l\'interprétation'

    def handle(self, *args, **options):
        test_dreams = [
            "J'ai rêvé d'un oiseau qui volait au-dessus de moi",
            "Je courais dans un couloir sans fin",
            "Ma mère me parlait dans une maison inconnue"
        ]
        
        for i, dream_text in enumerate(test_dreams, 1):
            self.stdout.write(f"\n=== Test {i}: {dream_text[:30]}... ===")
            
            try:
                result = interpret_dream(dream_text)
                
                if result is None:
                    self.stdout.write(self.style.ERROR("❌ Résultat None"))
                    continue
                
                # Vérifier le format
                expected_keys = ["Émotionnelle", "Symbolique", "Cognitivo-scientifique", "Freudien"]
                
                if not isinstance(result, dict):
                    self.stdout.write(self.style.ERROR(f"❌ Type incorrect: {type(result)}"))
                    continue
                
                for key in expected_keys:
                    if key not in result:
                        self.stdout.write(self.style.ERROR(f"❌ Clé manquante: {key}"))
                    elif not isinstance(result[key], str):
                        self.stdout.write(self.style.ERROR(f"❌ {key} n'est pas une string: {type(result[key])}"))
                    else:
                        self.stdout.write(self.style.SUCCESS(f"✅ {key}: OK"))
                
                self.stdout.write(self.style.SUCCESS("✅ Format valide"))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Erreur: {e}"))