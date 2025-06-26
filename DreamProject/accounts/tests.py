from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.contrib.messages import get_messages
from .forms import RegisterForm, LoginForm
from .models import CustomUser

User = get_user_model()


class CustomUserModelTests(TestCase):
    """Tester le custom user"""

    def set_up(self):
        self.user_data = {
            'email': 'test@gmail.com',
            'username': 'test_user',
            'password': 'Test_password123!',
        }

    def test_create_user(self):
        """Tester la création d'un user normal"""
        user = User.objects.create_user(
            email=self.user_data['email'],
            username=self.user_data['username'],
            password=self.user_data['password'],
        )

        self.assertEqual(user.email, self.user_data['email'])
        self.assertEqual(user.username, self.user_data['username'])
        self.assertTrue(user.check_password(self.user_data['password']))
        self.assertTrue(user.is_active)