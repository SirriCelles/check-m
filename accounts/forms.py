from django import forms

class RegistrationForm(forms.Form):
    first_name = forms.CharField(label='First Name', max_length=100)
    last_name = forms.CharField(label='Last Name', max_length=100)
    username = forms.CharField(label='UserName', max_length=100)
    email = forms.EmailField(label='Email' , max_length=100)
    password = forms.CharField(label='UserName', max_length=100)
    re_pass = forms.CharField(label='UserName', max_length=100)