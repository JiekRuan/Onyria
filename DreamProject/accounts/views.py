from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.urls import reverse_lazy
from .forms import RegisterForm, LoginForm, CustomPasswordChangeForm


def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('/diary/record/')        
        else:
            # Si le formulaire n'est pas valide, on le renvoie avec les erreurs
            return render(request, 'accounts/register.html', {'form': form})
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                return redirect('/diary/record/')
            else:
                # Si l'authentification échoue, on renvoie un message d'erreur
                form.add_error(None, "Email ou mot de passe incorrect")
                return render(request, 'accounts/login.html', {'form': form})
        else:
            # Si le formulaire n'est pas valide, on le renvoie avec les erreurs
            return render(request, 'accounts/login.html', {'form': form})
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', {'form': form})

def profil_view(request):
    return render(request, 'accounts/profil.html', {'user': request.user})

def logout_view(request):
    logout(request)
    return redirect('login')  # Redirige vers la page de login après la déconnexion

@login_required
def custom_password_change_view(request):
    if request.method == 'POST':
        form = CustomPasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            form.save()
            return redirect('password_change_done')
    else:
        form = CustomPasswordChangeForm(user=request.user)
    return render(request, 'accounts/change_password.html', {'form': form})


@login_required
def delete_account_view(request):
    if request.method == 'POST':
        user = request.user
        user.delete()
        logout(request)
        return redirect('login')  # Redirige vers la page d'accueil après la suppression du compte
    return render(request, 'accounts/delete_account.html')
