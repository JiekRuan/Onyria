from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.cache import never_cache
from .forms import RegisterForm, LoginForm, CustomPasswordChangeForm, UserUpdateForm, BioForm


@require_http_methods(["GET", "POST"])
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

@require_http_methods(["GET", "POST"])
@never_cache
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

@require_http_methods(["GET", "POST"])
@login_required
def account_management_view(request):
    if request.method == 'POST':
        # Vérifier s'il y a un fichier image uploadé
        if 'profile_picture' in request.FILES:
            uploaded_file = request.FILES['profile_picture']
            
            # Lire les bytes du fichier
            image_bytes = uploaded_file.read()
            
            # Déterminer le format
            file_extension = uploaded_file.name.split('.')[-1].lower()
            format_mapping = {
                'png': 'PNG',
                'jpg': 'JPEG',
                'jpeg': 'JPEG',
                'gif': 'GIF',
                'bmp': 'BMP'
            }
            image_format = format_mapping.get(file_extension, 'PNG')
            
            # Stocker en base64
            request.user.set_profile_picture_from_bytes(image_bytes, format=image_format)
            request.user.save()
            
            return redirect('account_management')
        
        # Traitement normal du formulaire pour les autres champs
        form = UserUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('account_management')
    else:
        form = UserUpdateForm(instance=request.user)

    return render(request, 'accounts/account_management.html', {'form': form, 'user': request.user})

@require_POST
@login_required
@never_cache
def logout_view(request):
    logout(request)
    return redirect('login')  # Redirige vers la page de login après la déconnexion

@require_http_methods(["GET", "POST"])
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

@require_http_methods(["GET", "POST"])
@login_required
def delete_account_view(request):
    if request.method == 'POST':
        user = request.user
        user.delete()
        logout(request)
        return redirect('login')  # Redirige vers la page d'accueil après la suppression du compte
    return render(request, 'accounts/delete_account.html')

@login_required
@require_POST
def edit_bio(request):
    form = BioForm(request.POST, instance=request.user)
    if form.is_valid():
        form.save()
    return redirect("dream_diary")