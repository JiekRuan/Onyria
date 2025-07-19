from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid
import os

def user_directory_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return f"profile_pics/user_{instance.id}/{filename}"

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    profile_picture = models.ImageField(
        upload_to=user_directory_path,
        null=True,
        blank=True,
        verbose_name="Photo de profil"
    )

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email
