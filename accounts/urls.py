from django.urls import path, re_path
from . import views

urlpatterns = [
    path('user/register/', views.Show_reg_page, name='reg_page'),
    path('create/', views.Register_user, name='register_user'),
    path('user/login/', views.Show_login_page, name="login_page"),
    path('login/', views.User_login, name="user_login"),
    path('logout/', views.User_logout, name="logout"),
    path('activate/email', views.Show_activate_email, name='acc_email'),
    re_path(r'^activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$',
        views.activate, name='activate'),
]