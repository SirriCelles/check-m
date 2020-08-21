from django.shortcuts import render
from django.contrib.auth.models import User, auth
from .models import  Patient

def Show_authUser_view(request):
    return render(request, 'patient/auth_patient.html', {})
