from django.urls import path
from .views import register_view, login_view, gestion_compte_view, logout_view, delete_account_view, custom_password_change_view
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path("my-account/", gestion_compte_view, name="gestion_compte"),
    path('logout/', logout_view, name='logout'),
    path('password-change/', custom_password_change_view, name='password_change'),
    path('password-change/done/', auth_views.PasswordChangeDoneView.as_view(template_name='accounts/password_change_done.html'), name='password_change_done'),
    path('delete-account/', delete_account_view, name='delete_account'),
]
