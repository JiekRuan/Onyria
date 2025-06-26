from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.views import PasswordChangeView
from django.urls import reverse_lazy
from .forms import RegisterForm, LoginForm

def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return render(request, 'profil.html', {'user': request.user})
        else:
            # Si le formulaire n'est pas valide, on le renvoie avec les erreurs
            return render(request, 'register.html', {'form': form})
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                return render(request, 'profil.html', {'user': request.user})
            else:
                # Si l'authentification échoue, on renvoie un message d'erreur
                form.add_error(None, "Email ou mot de passe incorrect")
                return render(request, 'login.html', {'form': form})
        else:
            # Si le formulaire n'est pas valide, on le renvoie avec les erreurs
            return render(request, 'login.html', {'form': form})
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def profil_view(request):
    return render(request, 'profil.html', {'user': request.user})

def logout_view(request):
    logout(request)
    return redirect('login')  # Redirige vers la page de login après la déconnexion

class CustomPasswordChangeView(PasswordChangeView):
    template_name = 'change_password.html'
    success_url = reverse_lazy('password_change_done')
