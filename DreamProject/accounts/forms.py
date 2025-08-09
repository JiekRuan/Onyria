from django import forms
from django.contrib.auth.forms import UserCreationForm , PasswordChangeForm
from .models import CustomUser

class RegisterForm(UserCreationForm):
    email = forms.EmailField(label="Adresse mail")
    username = forms.CharField(label="Nom d’utilisateur")
    password1 = forms.CharField(label="Mot de passe", widget=forms.PasswordInput)
    password2 = forms.CharField(label="Confirmation du mot de passe", widget=forms.PasswordInput)

    class Meta:
        model = CustomUser
        fields = ('email', 'username', 'password1', 'password2')


class LoginForm(forms.Form):
    email = forms.EmailField(label="Adresse mail")
    password = forms.CharField(label="Mot de passe", widget=forms.PasswordInput)

class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['profile_picture']
        labels = {
            'profile_picture': '', 
        }


class CustomPasswordChangeForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Personnalisation des labels
        self.fields['old_password'].label = "Ancien mot de passe"
        self.fields['new_password1'].label = "Nouveau mot de passe"
        self.fields['new_password2'].label = "Confirmez le nouveau mot de passe"


class BioForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ["bio"]
        widgets = {
            "bio": forms.TextInput(attrs={
                "class": "w-full rounded-xl border px-3 py-2",
                "placeholder": "Ta phrase onirique… (max 180 caractères)",
                "maxlength": 180
            })
        }

