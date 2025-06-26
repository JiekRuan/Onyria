from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

# Create your models here.


class Dream(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    transcription = models.TextField()
    date = models.DateTimeField('date du reve enregistré')
    image_base64 = models.TextField(
        verbose_name="Image en Base64",
        help_text="Image du rêve encodée en base64",
    )
    
    def __str__(self):
        return self.transcription
