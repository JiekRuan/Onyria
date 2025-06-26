from django.shortcuts import render
from django.shortcuts import render, get_object_or_404, reverse
from .models import Dream


from django.contrib.auth.decorators import login_required
# Create your views here.

@login_required
def index(request):
    dreams_history = Dream.objects.filter(user=request.user)
    context = {"List of dreams": dreams_history}
    return render(request, "diary/index.html", context)