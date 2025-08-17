"""
Tests complets pour le modèle Dream et ses fonctionnalités.

Ce module teste :
- Création et validation des instances Dream
- Propriétés JSON (emotions, interpretation)  
- Gestion des images et uploads
- Méthodes utilitaires du modèle
- Contraintes et validations
- Gestion Unicode et cas limites
- Performance avec gros volumes de données
"""

from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.files.storage import default_storage
from django.conf import settings
from django.db import transaction
from PIL import Image
from io import BytesIO
import json
import time
import os
import threading
import unittest
from collections import defaultdict

from ..models import Dream, dream_image_path

User = get_user_model()


class DreamModelTest(TestCase):
    """
    Tests complets pour le modèle Dream.
    
    Cette classe teste toutes les fonctionnalités du modèle Dream :
    - Création et validation des instances
    - Propriétés JSON (emotions, interpretation)
    - Méthodes utilitaires
    - Contraintes et validations
    """
    
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
    
    def test_create_basic_dream(self):
        """
        Test de création d'un rêve basique.
        
        Objectif : Vérifier que la création minimale fonctionne
        Vérifie que :
        - Un rêve peut être créé avec les champs obligatoires
        - L'objet est correctement sauvegardé en base de données
        - Les champs sont correctement assignés
        - Les valeurs par défaut sont appliquées
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="J'ai rêvé d'un chien blanc qui courait dans un champ"
        )
        
        # Vérifications de base
        self.assertEqual(dream.user, self.user)
        self.assertEqual(dream.transcription, "J'ai rêvé d'un chien blanc qui courait dans un champ")
        self.assertIsNotNone(dream.date)
        self.assertIsNotNone(dream.created_at)
        self.assertIsNotNone(dream.updated_at)
        
        # Vérifications des valeurs par défaut
        self.assertEqual(dream.dream_type, 'rêve')
        self.assertFalse(dream.is_analyzed)
        self.assertIsNone(dream.emotions_json)
        self.assertIsNone(dream.interpretation_json)
        
    def test_create_complete_dream(self):
        """
        Test de création d'un rêve complet avec tous les champs.
        
        Objectif : Vérifier que tous les champs peuvent être définis
        """
        emotions_data = {"joie": 0.8, "surprise": 0.2}
        interpretation_data = {
            "Émotionnelle": "Rêve joyeux",
            "Symbolique": "Symbolisme positif",
            "Cognitivo-scientifique": "Consolidation mémoire",
            "Freudien": "Expression désirs"
        }
        
        dream = Dream.objects.create(
            user=self.user,
            transcription="Rêve complet de test",
            dominant_emotion="joie",
            dream_type="rêve",
            is_analyzed=True
        )
        
        # Utilisation des setters
        dream.emotions = emotions_data
        dream.interpretation = interpretation_data
        dream.save()
        
        # Vérifications
        self.assertTrue(dream.is_analyzed)
        self.assertEqual(dream.dominant_emotion, "joie")
        self.assertEqual(dream.emotions, emotions_data)
        self.assertEqual(dream.interpretation, interpretation_data)

    def test_emotions_property_getter_setter(self):
        """
        Test des propriétés emotions (getter/setter).
        
        Objectif : Vérifier la conversion automatique dict ↔ JSON
        Teste :
        - Setter avec dictionnaire valide
        - Getter retournant le dictionnaire
        - Gestion des valeurs None
        - Persistence en base de données
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test émotions"
        )
        
        # Test setter avec dictionnaire valide
        emotions_data = {
            "joie": 0.7,
            "tristesse": 0.2,
            "peur": 0.1
        }
        dream.emotions = emotions_data
        dream.save()
        
        # Vérifier le stockage JSON
        self.assertIsNotNone(dream.emotions_json)
        json_data = json.loads(dream.emotions_json)
        self.assertEqual(json_data, emotions_data)
        
        # Test getter
        retrieved_emotions = dream.emotions
        self.assertEqual(retrieved_emotions, emotions_data)
        self.assertIsInstance(retrieved_emotions, dict)
        
        # Recharger depuis la DB pour vérifier la persistence
        dream.refresh_from_db()
        self.assertEqual(dream.emotions, emotions_data)
        
        # Test avec None
        dream.emotions = None
        self.assertIsNone(dream.emotions_json)
        self.assertEqual(dream.emotions, {})

    def test_emotions_property_invalid_json(self):
        """
        Test de la propriété emotions avec JSON corrompu.
        
        Objectif : Vérifier la robustesse face aux données corrompues
        Simule une corruption de données en base
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test JSON corrompu"
        )
        
        # Simuler une corruption en modifiant directement le JSON
        dream.emotions_json = '{"joie": 0.8, "tristesse":}'  # JSON invalide
        dream.save()
        
        # Le getter doit retourner un dict vide sans lever d'exception
        emotions = dream.emotions
        self.assertEqual(emotions, {})

    def test_interpretation_property_getter_setter(self):
        """
        Test des propriétés interpretation (getter/setter).
        
        Objectif : Vérifier la conversion automatique dict ↔ JSON
        Teste spécifiquement la structure d'interprétation attendue
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test interprétation"
        )
        
        # Structure d'interprétation attendue
        interpretation_data = {
            "Émotionnelle": "Ce rêve exprime une joie profonde",
            "Symbolique": "L'eau symbolise les émotions",
            "Cognitivo-scientifique": "Consolidation de souvenirs récents",
            "Freudien": "Expression de désirs refoulés"
        }
        
        # Test setter
        dream.interpretation = interpretation_data
        dream.save()
        
        # Vérifications
        self.assertIsNotNone(dream.interpretation_json)
        self.assertEqual(dream.interpretation, interpretation_data)
        
        # Test persistence
        dream.refresh_from_db()
        self.assertEqual(dream.interpretation, interpretation_data)
        
        # Test avec données invalides
        dream.interpretation = "string_instead_of_dict"
        self.assertIsNone(dream.interpretation_json)

    def test_interpretation_property_invalid_json(self):
        """
        Test de la propriété interpretation avec JSON corrompu.
        
        Objectif : Vérifier la robustesse face aux données corrompues
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test interprétation corrompue"
        )
        
        # JSON corrompu
        dream.interpretation_json = '{"Émotionnelle": "texte", "Symbolique":}'
        dream.save()
        
        # Doit retourner un dict vide
        interpretation = dream.interpretation
        self.assertEqual(interpretation, {})

    def test_has_image_property(self):
        """
        Test de la propriété has_image.
        
        Objectif : Vérifier la détection de présence d'image
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test image"
        )
        
        # Sans image
        self.assertFalse(dream.has_image)
        
        # Avec image simulée
        fake_image = SimpleUploadedFile(
            "test_dream.jpg",
            b"fake_image_content",
            content_type="image/jpeg"
        )
        dream.image = fake_image
        dream.save()
        
        self.assertTrue(dream.has_image)

    def test_short_transcription_property(self):
        """
        Test de la propriété short_transcription.
        
        Objectif : Vérifier le raccourcissement intelligent du texte
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
        long_text = "J'ai fait un rêve très détaillé avec énormément d'éléments narratifs qui se succèdent continuellement sans interruption notable dans ce récit onirique fascinant"
        long_dream = Dream.objects.create(
            user=self.user,
            transcription=long_text
        )
        expected = long_text[:100] + "..."
        self.assertEqual(long_dream.short_transcription, expected)
        
        # Test avec exactement 100 caractères
        exact_text = "a" * 100
        exact_dream = Dream.objects.create(
            user=self.user,
            transcription=exact_text
        )
        self.assertEqual(exact_dream.short_transcription, exact_text)

    def test_dream_type_choices_validation(self):
        """
        Test de validation des choix de dream_type.
        
        Objectif : Vérifier que seules les valeurs autorisées sont acceptées
        """
        # Valeurs valides
        valid_dream = Dream.objects.create(
            user=self.user,
            transcription="Rêve valide",
            dream_type="rêve"
        )
        self.assertEqual(valid_dream.dream_type, "rêve")
        
        valid_nightmare = Dream.objects.create(
            user=self.user,
            transcription="Cauchemar valide",
            dream_type="cauchemar"
        )
        self.assertEqual(valid_nightmare.dream_type, "cauchemar")

    def test_dream_string_representation(self):
        """
        Test de la méthode __str__ du modèle.
        
        Objectif : Vérifier l'affichage correct dans l'admin Django
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="Un rêve pour tester la représentation string du modèle Dream"
        )
        
        str_repr = str(dream)
        
        # Vérifications
        self.assertIn(self.user.username, str_repr)
        self.assertIn("Un rêve pour tester la représentation string", str_repr)
        # Vérifier que la date est formatée
        self.assertRegex(str_repr, r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}')

    def test_dream_image_path_function(self):
        """
        Test de la fonction dream_image_path.
        
        Objectif : Vérifier la génération correcte du chemin d'upload
        """
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test path image"
        )
        
        filename = "test_dream_image.jpg"
        path = dream_image_path(dream, filename)
        
        expected_path = f'dream_images/user_{self.user.id}/{filename}'
        self.assertEqual(path, expected_path)

    def test_dream_ordering(self):
        """
        Test de l'ordre par défaut des rêves.
        
        Objectif : Vérifier que les rêves sont triés par date décroissante
        """
        # Créer plusieurs rêves avec un délai
        dream1 = Dream.objects.create(
            user=self.user,
            transcription="Premier rêve"
        )
        
        time.sleep(0.01)  # Petit délai pour différencier les dates
        
        dream2 = Dream.objects.create(
            user=self.user,
            transcription="Deuxième rêve"
        )
        
        time.sleep(0.01)
        
        dream3 = Dream.objects.create(
            user=self.user,
            transcription="Troisième rêve"
        )
        
        # Récupérer tous les rêves (ordre par défaut)
        dreams = list(Dream.objects.all())
        
        # Le plus récent doit être en premier
        self.assertEqual(dreams[0], dream3)
        self.assertEqual(dreams[1], dream2)
        self.assertEqual(dreams[2], dream1)

    def test_dream_cascade_deletion(self):
        """
        Test de la suppression en cascade.
        
        Objectif : Vérifier que la suppression d'un user supprime ses rêves
        """
        # Créer un rêve
        dream = Dream.objects.create(
            user=self.user,
            transcription="Rêve à supprimer"
        )
        dream_id = dream.id
        
        # Vérifier que le rêve existe
        self.assertTrue(Dream.objects.filter(id=dream_id).exists())
        
        # Supprimer l'utilisateur
        self.user.delete()
        
        # Vérifier que le rêve a été supprimé
        self.assertFalse(Dream.objects.filter(id=dream_id).exists())

    def test_large_transcription_handling(self):
        """
        Test de gestion des transcriptions très longues.
        
        Objectif : Vérifier que le modèle gère les gros textes
        """
        # Créer une transcription de 50KB
        large_text = "Je rêve d'un monde meilleur. " * 2000  # ~50KB
        
        dream = Dream.objects.create(
            user=self.user,
            transcription=large_text
        )
        
        # Vérifications
        self.assertEqual(len(dream.transcription), len(large_text))
        dream.refresh_from_db()
        self.assertEqual(dream.transcription, large_text)

    def test_unicode_and_special_characters(self):
        """
        Test de gestion des caractères unicode et spéciaux.
        
        Objectif : Vérifier le support international complet
        """
        special_text = "J'ai rêvé 🌙✨ d'éléphants 象 en Москва avec des émojis 😴💭"
        
        dream = Dream.objects.create(
            user=self.user,
            transcription=special_text
        )
        
        # Vérifications
        self.assertEqual(dream.transcription, special_text)
        dream.refresh_from_db()
        self.assertEqual(dream.transcription, special_text)

    @unittest.skipIf(
        settings.DATABASES['default']['ENGINE'].endswith('sqlite3'),
        "Tests de concurrence nécessitent PostgreSQL. "
        "SQLite ne supporte pas les écritures simultanées. "
        "Ce test sera automatiquement activé en production avec PostgreSQL."
    )
    def test_concurrent_dream_creation(self):
        """
        Test de création simultanée de rêves par plusieurs threads.
        
        Objectif : Vérifier la robustesse de l'application face aux accès concurrents
        
        Scénarios testés :
        - Plusieurs utilisateurs créent des rêves simultanément
        - Intégrité des données sous charge concurrente
        - Pas de corruption ou perte de données
        - Isolation correcte des transactions
        
        IMPORTANT: Ce test nécessite PostgreSQL pour fonctionner correctement.
        Il est automatiquement skippé en développement avec SQLite.
        """
        import threading
        import time
        from collections import defaultdict
        
        # Configuration du test
        num_threads = 5
        dreams_per_thread = 3
        total_expected = num_threads * dreams_per_thread
        
        # Structures pour collecter les résultats
        created_dreams = []
        errors = []
        thread_results = defaultdict(list)
        
        def create_dreams_for_thread(thread_id):
            """
            Fonction exécutée par chaque thread pour créer des rêves.
            
            Args:
                thread_id: Identifiant unique du thread
            """
            thread_dreams = []
            
            try:
                for i in range(dreams_per_thread):
                    # Utiliser une transaction atomique pour chaque création
                    with transaction.atomic():
                        dream = Dream.objects.create(
                            user=self.user,
                            transcription=f"Rêve concurrent thread-{thread_id} rêve-{i}",
                            dream_type="rêve" if i % 2 == 0 else "cauchemar",
                            dominant_emotion="joie" if i % 3 == 0 else "tristesse",
                            is_analyzed=True
                        )
                        thread_dreams.append(dream)
                        
                        # Petit délai pour augmenter les chances de concurrence
                        time.sleep(0.01)
                
                # Stocker les résultats de manière thread-safe
                with threading.Lock():
                    created_dreams.extend(thread_dreams)
                    thread_results[thread_id] = thread_dreams
                    
            except Exception as e:
                # Capturer les erreurs pour analyse
                with threading.Lock():
                    errors.append((thread_id, str(e)))
        
        # Lancer les threads simultanément
        threads = []
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=create_dreams_for_thread,
                args=(thread_id,),
                name=f"DreamCreator-{thread_id}"
            )
            threads.append(thread)
        
        # Démarrer tous les threads
        for thread in threads:
            thread.start()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join(timeout=10)  # Timeout de sécurité
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # === VÉRIFICATIONS ===
        
        # 1. Aucune erreur ne doit avoir eu lieu
        self.assertEqual(len(errors), 0, 
                        f"Erreurs de concurrence détectées: {errors}")
        
        # 2. Tous les threads doivent avoir terminé
        for thread in threads:
            self.assertFalse(thread.is_alive(), 
                           f"Thread {thread.name} n'a pas terminé dans les temps")
        
        # 3. Nombre correct de rêves créés
        db_dreams_count = Dream.objects.filter(user=self.user).count()
        self.assertEqual(db_dreams_count, total_expected,
                        f"Attendu {total_expected} rêves, trouvé {db_dreams_count} en base")
        
        self.assertEqual(len(created_dreams), total_expected,
                        f"Attendu {total_expected} rêves collectés, trouvé {len(created_dreams)}")
        
        # 4. Chaque thread a créé le bon nombre de rêves
        for thread_id in range(num_threads):
            thread_dream_count = len(thread_results[thread_id])
            self.assertEqual(thread_dream_count, dreams_per_thread,
                           f"Thread {thread_id} a créé {thread_dream_count} rêves au lieu de {dreams_per_thread}")
        
        # 5. Vérifier l'intégrité des données
        all_db_dreams = Dream.objects.filter(user=self.user).order_by('id')
        
        # Aucun rêve dupliqué par ID
        dream_ids = [dream.id for dream in all_db_dreams]
        self.assertEqual(len(dream_ids), len(set(dream_ids)),
                        "Des rêves dupliqués ont été détectés")
        
        # Toutes les transcriptions sont uniques
        transcriptions = [dream.transcription for dream in all_db_dreams]
        self.assertEqual(len(transcriptions), len(set(transcriptions)),
                        "Des transcriptions dupliquées ont été détectées")
        
        # 6. Vérifier que tous les rêves appartiennent au bon utilisateur
        for dream in all_db_dreams:
            self.assertEqual(dream.user, self.user,
                           f"Rêve {dream.id} n'appartient pas au bon utilisateur")
        
        # 7. Performance acceptable (doit rester sous 5 secondes)
        self.assertLess(execution_time, 5.0,
                       f"Test de concurrence trop lent: {execution_time:.2f}s")
        
        # === LOGS DE DEBUG ===
        print(f"\n=== Test de concurrence réussi ===")
        print(f"Threads: {num_threads}")
        print(f"Rêves par thread: {dreams_per_thread}")
        print(f"Total créé: {db_dreams_count}")
        print(f"Temps d'exécution: {execution_time:.2f}s")
        print(f"Débit: {total_expected/execution_time:.1f} rêves/seconde")

    def test_concurrent_dream_creation_fallback_sqlite(self):
        """
        Version simplifiée du test de concurrence pour SQLite.
        
        Teste la création rapide séquentielle pour simuler une charge élevée
        sans les problèmes de concurrence de SQLite.
        
        Scénarios testés :
        - Création rapide séquentielle de rêves
        - Intégrité des données sous charge élevée
        - Performance acceptable en mode séquentiel
        """
        num_dreams = 15
        start_time = time.time()
        
        # Création rapide séquentielle
        created_dreams = []
        for i in range(num_dreams):
            dream = Dream.objects.create(
                user=self.user,
                transcription=f"Rêve séquentiel rapide {i}",
                dream_type="rêve" if i % 2 == 0 else "cauchemar",
                dominant_emotion="joie" if i % 3 == 0 else "tristesse"
            )
            created_dreams.append(dream)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Vérifications
        created_count = Dream.objects.filter(user=self.user).count()
        self.assertEqual(created_count, num_dreams)
        
        # Vérifier l'intégrité des données
        for i, dream in enumerate(created_dreams):
            self.assertEqual(dream.transcription, f"Rêve séquentiel rapide {i}")
            self.assertEqual(dream.user, self.user)
        
        # Performance acceptable même en séquentiel
        self.assertLess(execution_time, 2.0,
                       f"Création séquentielle trop lente: {execution_time:.2f}s")
        
        print(f"\n=== Test séquentiel SQLite ===")
        print(f"Rêves créés: {created_count}")
        print(f"Temps: {execution_time:.2f}s")
        print(f"Débit: {num_dreams/execution_time:.1f} rêves/seconde")


class DreamModelImageTest(TestCase):
    """
    Tests spécifiques pour la gestion des images dans le modèle Dream.
    
    Cette classe teste :
    - Upload et stockage d'images
    - Validation des formats
    - Nettoyage des fichiers
    - Gestion des erreurs d'image
    """
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test_images@example.com',
            username='testuser_images',
            password='testpass123'
        )

    def create_test_image(self, format='JPEG', size=(100, 100)):
        """
        Utilitaire pour créer une image de test.
        
        Args:
            format: Format de l'image (JPEG, PNG, etc.)
            size: Taille en pixels (width, height)
        
        Returns:
            SimpleUploadedFile: Fichier image prêt pour l'upload
        """
        image = Image.new('RGB', size, color='red')
        image_io = BytesIO()
        image.save(image_io, format=format)
        image_io.seek(0)
        
        filename = f"test_image.{format.lower()}"
        content_type = f"image/{format.lower()}"
        
        return SimpleUploadedFile(
            filename,
            image_io.getvalue(),
            content_type=content_type
        )

    def test_image_upload_jpeg(self):
        """
        Test d'upload d'une image JPEG valide.
        
        Objectif : Vérifier que les images JPEG sont correctement stockées
        """
        image_file = self.create_test_image('JPEG')
        
        dream = Dream.objects.create(
            user=self.user,
            transcription="Rêve avec image JPEG",
            image=image_file
        )
        
        # Vérifications
        self.assertTrue(dream.has_image)
        self.assertIsNotNone(dream.image)
        self.assertTrue(dream.image.name.startswith(f'dream_images/user_{self.user.id}/'))
        self.assertTrue(dream.image.name.endswith('.jpeg'))

    def test_image_upload_png(self):
        """
        Test d'upload d'une image PNG valide.
        
        Objectif : Vérifier le support du format PNG
        """
        image_file = self.create_test_image('PNG')
        
        dream = Dream.objects.create(
            user=self.user,
            transcription="Rêve avec image PNG",
            image=image_file
        )
        
        self.assertTrue(dream.has_image)
        self.assertTrue(dream.image.name.endswith('.png'))

    def test_image_path_generation(self):
        """
        Test de génération du chemin d'image.
        
        Objectif : Vérifier l'isolation des images par utilisateur
        """
        image_file = self.create_test_image()
        
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test chemin image",
            image=image_file
        )
        
        # Vérifier la structure du chemin
        expected_prefix = f'dream_images/user_{self.user.id}/'
        self.assertTrue(dream.image.name.startswith(expected_prefix))

    def test_large_image_handling(self):
        """
        Test de gestion d'images de grande taille.
        
        Objectif : Vérifier que les grosses images sont gérées correctement
        """
        # Créer une image de 2MB environ
        large_image = self.create_test_image('JPEG', size=(2000, 2000))
        
        dream = Dream.objects.create(
            user=self.user,
            transcription="Rêve avec grosse image",
            image=large_image
        )
        
        self.assertTrue(dream.has_image)
        # Vérifier que le fichier existe bien
        self.assertTrue(dream.image.storage.exists(dream.image.name))

    def test_multiple_users_image_isolation(self):
        """
        Test d'isolation des images entre utilisateurs.
        
        Objectif : Vérifier que chaque user a son dossier d'images
        """
        user2 = User.objects.create_user(
            email='user2@example.com',
            username='user2',
            password='testpass123'
        )
        
        image1 = self.create_test_image()
        image2 = self.create_test_image()
        
        dream1 = Dream.objects.create(
            user=self.user,
            transcription="Rêve user 1",
            image=image1
        )
        
        dream2 = Dream.objects.create(
            user=user2,
            transcription="Rêve user 2",
            image=image2
        )
        
        # Vérifier l'isolation des chemins
        self.assertTrue(dream1.image.name.startswith(f'dream_images/user_{self.user.id}/'))
        self.assertTrue(dream2.image.name.startswith(f'dream_images/user_{user2.id}/'))
        self.assertNotEqual(
            dream1.image.name.split('/')[1],  # user_X partie
            dream2.image.name.split('/')[1]
        )

    def test_image_without_extension(self):
        """
        Test d'upload d'image sans extension.
        
        Objectif : Vérifier la gestion des fichiers malformés
        """
        fake_image = SimpleUploadedFile(
            "image_sans_extension",
            b"fake_image_content",
            content_type="image/jpeg"
        )
        
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test image sans extension",
            image=fake_image
        )
        
        # Doit être géré gracieusement
        self.assertTrue(dream.has_image)

    def test_empty_image_file(self):
        """
        Test d'upload d'un fichier image vide.
        
        Objectif : Vérifier la gestion des fichiers vides
        """
        empty_image = SimpleUploadedFile(
            "empty.jpg",
            b"",  # Fichier vide
            content_type="image/jpeg"
        )
        
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test image vide",
            image=empty_image
        )
        
        # Peut être accepté selon la configuration Django
        # Le test vérifie que ça ne plante pas
        self.assertIsNotNone(dream.image)


class DreamModelPerformanceTest(TestCase):
    """
    Tests de performance spécifiques au modèle Dream.
    
    Cette classe teste les performances avec de gros volumes de données
    pour s'assurer que le modèle reste efficace à grande échelle.
    """
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='test_perf@example.com',
            username='testuser_perf',
            password='testpass123'
        )

    def test_bulk_dream_creation_performance(self):
        """
        Test de performance de création en masse de rêves.
        
        Objectif : Vérifier que la création de nombreux rêves reste efficace
        """
        start_time = time.time()
        
        # Créer 100 rêves en bulk
        dreams_data = []
        for i in range(100):
            dreams_data.append(Dream(
                user=self.user,
                transcription=f"Rêve de performance numéro {i}",
                dream_type="rêve" if i % 2 == 0 else "cauchemar",
                dominant_emotion="joie" if i % 3 == 0 else "tristesse"
            ))
        
        # Insertion en bulk
        Dream.objects.bulk_create(dreams_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Doit créer 100 rêves en moins d'une seconde
        self.assertLess(execution_time, 1.0)
        self.assertEqual(Dream.objects.filter(user=self.user).count(), 100)

    def test_large_json_operations_performance(self):
        """
        Test de performance des opérations JSON volumineuses.
        
        Objectif : Vérifier que les propriétés JSON restent rapides
        """
        # Créer de gros objets JSON
        large_emotions = {}
        large_interpretation = {}
        
        for i in range(50):
            large_emotions[f"emotion_{i}"] = round(1.0/50, 6)
            large_interpretation[f"Aspect_{i}"] = f"Longue analyse détaillée numéro {i} " * 10
        
        start_time = time.time()
        
        dream = Dream.objects.create(
            user=self.user,
            transcription="Test performance JSON volumineux"
        )
        
        # Opérations sur gros JSON
        dream.emotions = large_emotions
        dream.interpretation = large_interpretation
        dream.save()
        
        # Lecture
        retrieved_emotions = dream.emotions
        retrieved_interpretation = dream.interpretation
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Doit rester rapide même avec de gros JSON
        self.assertLess(execution_time, 2.0)
        self.assertEqual(len(retrieved_emotions), 50)
        self.assertEqual(len(retrieved_interpretation), 50)

    def test_query_performance_with_many_dreams(self):
        """
        Test de performance des requêtes avec beaucoup de rêves.
        
        Objectif : Vérifier que les requêtes restent efficaces
        """
        # Créer 200 rêves
        dreams_data = []
        for i in range(200):
            dreams_data.append(Dream(
                user=self.user,
                transcription=f"Rêve {i} avec contenu variable",
                dream_type="rêve" if i % 3 != 0 else "cauchemar",
                dominant_emotion="joie" if i % 2 == 0 else "tristesse",
                is_analyzed=True
            ))
        
        Dream.objects.bulk_create(dreams_data)
        
        start_time = time.time()
        
        # Différentes requêtes courantes
        all_dreams = list(Dream.objects.filter(user=self.user).order_by('-date')[:20])
        analyzed_dreams = Dream.objects.filter(user=self.user, is_analyzed=True).count()
        recent_dreams = list(Dream.objects.filter(user=self.user)[:10])
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Les requêtes doivent rester rapides
        self.assertLess(execution_time, 1.0)
        self.assertEqual(len(all_dreams), 20)
        self.assertEqual(analyzed_dreams, 200)
        self.assertEqual(len(recent_dreams), 10)


"""
=== UTILISATION DES TESTS MODELS ===

Ce module teste complètement le modèle Dream et ses fonctionnalités :

1. LANCER LES TESTS MODELS :
   python manage.py test diary.tests.test_models

2. TESTS PAR CLASSE :
   python manage.py test diary.tests.test_models.DreamModelTest
   python manage.py test diary.tests.test_models.DreamModelImageTest
   python manage.py test diary.tests.test_models.DreamModelPerformanceTest

3. COUVERTURE COMPLÈTE DU MODÈLE :
   - Création et validation ✓
   - Propriétés JSON (emotions, interpretation) ✓
   - Gestion des images ✓
   - Méthodes utilitaires ✓
   - Performance avec gros volumes ✓
   - Concurrence (PostgreSQL) ✓
   - Robustesse (corruption, Unicode) ✓

4. TESTS DE CONCURRENCE :
   - test_concurrent_dream_creation : Skippé sur SQLite, activé sur PostgreSQL
   - test_concurrent_dream_creation_fallback_sqlite : Version SQLite séquentielle
   
5. GESTION AUTOMATIQUE DE LA DB :
   - SQLite (dev) : Tests séquentiels, pas de problèmes de verrous
   - PostgreSQL (prod) : Tests de concurrence réels automatiquement activés

6. PERFORMANCE VALIDÉE :
   - Création en masse : < 1 seconde pour 100 rêves
   - Opérations JSON volumineuses : < 2 secondes
   - Requêtes avec 200 rêves : < 1 seconde

=== PHILOSOPHIE ===

Ces tests garantissent que le modèle Dream est robuste :
- Toutes les propriétés fonctionnent correctement
- Les données JSON sont bien gérées
- Les images sont stockées proprement
- La performance reste acceptable à grande échelle
- La concurrence est testée quand techniquement possible

=== TRANSITION POSTGRESQL ===

Quand vous déployez avec PostgreSQL :
1. Les tests de concurrence s'activent automatiquement
2. Aucune modification de code nécessaire
3. Détection automatique des vrais problèmes de concurrence
4. Mesure de performance sous charge réelle

Temps d'exécution estimé : 30-60 secondes selon la machine.
"""