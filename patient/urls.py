from django.urls import path

from . import views

urlpatterns = [
    path('auth/user/', views.Show_authUser_view, name='patient_view'),
]