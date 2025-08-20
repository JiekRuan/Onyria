from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import PasswordChangeForm
from .forms import RegisterForm, LoginForm, CustomPasswordChangeForm
from .models import CustomUser


User = get_user_model()


class CustomUserModelTest(TestCase):
    def test_create_user(self):
        user = User.objects.create_user(email="test@example.com", username="testuser", password="testpass123")
        self.assertEqual(user.email, "test@example.com")
        self.assertTrue(user.check_password("testpass123"))

    def test_create_user_with_age_and_sexe(self):
        user = User.objects.create_user(
            email="test@example.com", 
            username="testuser", 
            password="testpass123",
            age=25,
            sexe='F'
        )
        self.assertEqual(user.age, 25)
        self.assertEqual(user.sexe, 'F')

    def test_user_age_optional(self):
        user = User.objects.create_user(
            email="test@example.com", 
            username="testuser", 
            password="testpass123"
        )
        self.assertIsNone(user.age)
        self.assertIsNone(user.sexe)

    def test_gender_choices(self):
        choices = [choice[0] for choice in CustomUser.GENDER_CHOICES]
        self.assertIn('M', choices)
        self.assertIn('F', choices)
        self.assertIn('O', choices)
        self.assertIn('N', choices)


class RegisterFormTest(TestCase):
    def test_valid_register_form(self):
        form_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
        }
        form = RegisterForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_valid_register_form_with_age_and_sexe(self):
        form_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
            "age": 25,
            "sexe": "F",
        }
        form = RegisterForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_register_form_age_too_young(self):
        form_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
            "age": 12,
        }
        form = RegisterForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("Ensure this value is greater than or equal to 13", str(form.errors['age']))

    def test_register_form_age_too_old(self):
        form_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
            "age": 121,
        }
        form = RegisterForm(data=form_data)
        self.assertFalse(form.is_valid())

    def test_register_form_age_optional(self):
        form_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
            "age": "",  # Age vide
        }
        form = RegisterForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_register_form_sexe_optional(self):
        form_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
            "sexe": "",  # Sexe vide
        }
        form = RegisterForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_register_form_invalid_sexe_choice(self):
        form_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
            "sexe": "X",  # Choix invalide
        }
        form = RegisterForm(data=form_data)
        self.assertFalse(form.is_valid())

    def test_invalid_register_form_password_mismatch(self):
        form_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password1": "ComplexPwd123!",
            "password2": "WrongPwd123!",
        }
        form = RegisterForm(data=form_data)
        self.assertFalse(form.is_valid())


class LoginFormTest(TestCase):
    def test_valid_login_form(self):
        form_data = {
            "email": "test@example.com",
            "password": "secret",
        }
        form = LoginForm(data=form_data)
        self.assertTrue(form.is_valid())


class PasswordChangeFormTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", username="user", password="oldpass123")
        self.client.login(email="user@example.com", password="oldpass123")

    def test_custom_password_change_form_labels(self):
        form = CustomPasswordChangeForm(user=self.user)
        self.assertEqual(form.fields["old_password"].label, "Ancien mot de passe")
        self.assertEqual(form.fields["new_password1"].label, "Nouveau mot de passe")
        self.assertEqual(form.fields["new_password2"].label, "Confirmez le nouveau mot de passe")


class ViewsTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(email="user@example.com", username="user", password="userpass123")

    def test_register_view_get(self):
        response = self.client.get(reverse("register"))
        self.assertEqual(response.status_code, 200)

    def test_register_view_post_with_age_and_sexe(self):
        response = self.client.post(reverse("register"), {
            "email": "newuser@example.com",
            "username": "newuser",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
            "age": 25,
            "sexe": "M",
        })
        self.assertEqual(response.status_code, 302)  # Redirection après inscription
        
        # Vérifier que l'utilisateur a été créé avec les bonnes données
        user = User.objects.get(email="newuser@example.com")
        self.assertEqual(user.age, 25)
        self.assertEqual(user.sexe, "M")

    def test_register_view_post_without_optional_fields(self):
        response = self.client.post(reverse("register"), {
            "email": "newuser2@example.com",
            "username": "newuser2",
            "password1": "ComplexPwd123!",
            "password2": "ComplexPwd123!",
        })
        self.assertEqual(response.status_code, 302)
        
        # Vérifier que l'utilisateur a été créé sans âge ni sexe
        user = User.objects.get(email="newuser2@example.com")
        self.assertIsNone(user.age)
        self.assertEqual(user.sexe, "")  # CharField vide retourne une chaîne vide

    def test_login_view_get(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)

    def test_login_view_post_valid(self):
        response = self.client.post(reverse("login"), {
            "email": "user@example.com",
            "password": "userpass123"
        })
        self.assertRedirects(response, "/diary/record/")

    def test_logout_view(self):
        self.client.login(email="user@example.com", password="userpass123")
        response = self.client.post(reverse("logout"))  # POST au lieu de GET
        self.assertEqual(response.status_code, 302)

    def test_delete_account_view_post(self):
        self.client.login(email="user@example.com", password="userpass123")
        response = self.client.post(reverse("delete_account"))
        self.assertRedirects(response, reverse("login"))