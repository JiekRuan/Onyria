from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

class RegisterForm(UserCreationForm):
    email = forms.EmailField(max_length=254, help_text='Required. Valid email address.')

    class Meta:
        model = CustomUser
        fields = ('email', 'username', 'password1', 'password2')

class LoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)