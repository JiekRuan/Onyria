from django.db import models
from django.contrib.auth.models import AbstractUser
import base64

class CustomUser(AbstractUser):
    GENDER_CHOICES = [
        ('M', 'Homme'),
        ('F', 'Femme'),
        ('O', 'Autre'),
        ('N', 'Préfère ne pas dire'),
    ]
    
    email = models.EmailField(unique=True)
    
    age = models.PositiveIntegerField(
        "Âge",
        null=True,
        blank=True,
        help_text="Votre âge"
    )
    
    sexe = models.CharField(
        "Sexe",
        max_length=1,
        choices=GENDER_CHOICES,
        blank=True,
        null=True,
        help_text="Votre sexe"
    )
    
    profile_picture_base64 = models.TextField(
        blank=True,
        null=True,
        verbose_name="Photo de profil (base64)",
        help_text="Photo de profil encodée en base64"
    )
    
    bio = models.CharField(
        "Bio",
        max_length=180,
        blank=True,
        default="",
        help_text="Une courte phrase (max 180 caractères)."
    )

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email

    def set_profile_picture_from_bytes(self, image_bytes, format='PNG'):
        """Encode une image de profil en base64 et la stocke"""
        if image_bytes:
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            mime_type = f"image/{format.lower()}"
            if format.upper() == 'JPG':
                mime_type = "image/jpeg"
            self.profile_picture_base64 = f"data:{mime_type};base64,{base64_string}"

    @property
    def has_profile_picture(self):
        """Vérifie si l'utilisateur a une photo de profil"""
        return bool(self.profile_picture_base64)

    @property
    def profile_picture_url(self):
        """Retourne l'URL de la photo de profil en base64"""
        return self.profile_picture_base64