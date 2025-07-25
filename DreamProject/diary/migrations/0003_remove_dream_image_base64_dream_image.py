# Generated by Django 4.2.18 on 2025-07-22 13:20

import diary.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diary', '0002_alter_dream_options_dream_created_at_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='dream',
            name='image_base64',
        ),
        migrations.AddField(
            model_name='dream',
            name='image',
            field=models.ImageField(blank=True, help_text='Image générée à partir du rêve', null=True, upload_to=diary.models.dream_image_path, verbose_name='Image du rêve'),
        ),
    ]
