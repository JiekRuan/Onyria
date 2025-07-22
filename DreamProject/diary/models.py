from django.db import models
from django.conf import settings
import json


def dream_image_path(instance, filename):
    return f'dream_images/user_{instance.user.id}/{filename}'


class Dream(models.Model):
    DREAM_TYPES = [
        ('rêve', 'Rêve'),
        ('cauchemar', 'Cauchemar'),
    ]

    # Informations de base
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE
    )
    transcription = models.TextField(verbose_name="Transcription du rêve")
    date = models.DateTimeField('Date du rêve enregistré', auto_now_add=True)

    # Analyse émotionnelle
    emotions_json = models.TextField(
        blank=True,
        null=True,
        verbose_name="Émotions (JSON)",
        help_text="Analyse des émotions au format JSON",
    )
    dominant_emotion = models.CharField(
        max_length=50, blank=True, null=True, verbose_name="Émotion dominante"
    )
    dream_type = models.CharField(
        max_length=10,
        choices=DREAM_TYPES,
        default='rêve',
        verbose_name="Type de rêve",
    )

    # Contenu généré
    image = models.ImageField(
        upload_to=dream_image_path,
        null=True,
        blank=True,
        verbose_name="Image du rêve",
        help_text="Image générée à partir du rêve"
    )
    image_prompt = models.TextField(
        blank=True,
        null=True,
        verbose_name="Prompt de l'image",
        help_text="Prompt utilisé pour générer l'image",
    )
    interpretation_json = models.TextField(
        blank=True,
        null=True,
        verbose_name="Interprétation (JSON)",
        help_text="Interprétation du rêve au format JSON",
    )

    # Métadonnées
    is_analyzed = models.BooleanField(
        default=False,
        verbose_name="Rêve analysé",
        help_text="Indique si l'analyse complète a été effectuée",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date']
        verbose_name = "Rêve"
        verbose_name_plural = "Rêves"

    def __str__(self):
        return f"{self.user.username} - {self.date.strftime('%d/%m/%Y %H:%M')} - {self.transcription[:50]}..."

    @property
    def emotions(self):
        """Retourne les émotions sous forme de dictionnaire"""
        if self.emotions_json:
            try:
                return json.loads(self.emotions_json)
            except json.JSONDecodeError:
                return {}
        return {}

    @emotions.setter
    def emotions(self, value):
        """Définit les émotions à partir d'un dictionnaire"""
        if isinstance(value, dict):
            self.emotions_json = json.dumps(value)
        else:
            self.emotions_json = None

    @property
    def interpretation(self):
        """Retourne l'interprétation sous forme de dictionnaire"""
        if self.interpretation_json:
            try:
                return json.loads(self.interpretation_json)
            except json.JSONDecodeError:
                return {}
        return {}

    @interpretation.setter
    def interpretation(self, value):
        """Définit l'interprétation à partir d'un dictionnaire"""
        if isinstance(value, dict):
            self.interpretation_json = json.dumps(value)
        else:
            self.interpretation_json = None

    @property
    def has_image(self):
        """Vérifie si le rêve a une image"""
        return bool(self.image)

    @property
    def short_transcription(self):
        """Retourne une version courte de la transcription"""
        if len(self.transcription) > 100:
            return self.transcription[:100] + "..."
        return self.transcription
