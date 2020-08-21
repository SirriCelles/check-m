import sweetify

from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
# from django.core.mail import EmailMessage

from django.core.mail import send_mail
from django.conf import settings
from .tokens import account_activation_token
# from django_email_verification import sendConfirm


# Create your views here.

def Show_reg_page(request):
    return render(request, 'register.html', {})

def Show_login_page(request):
    return render(request, 'login.html', {})

def login(request, user):
    auth.login(request)
    return redirect('/')

def Show_activate_email(request):
    return render(request, 'confirm_template.html', {})


def Register_user(request):
    response_data = {}
    if request.is_ajax() and request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        re_password = request.POST.get('re_password')
        
        
        if password == re_password:
            if User.objects.filter(username=username).exists():
                response_data['text'] = "Username already exist."
                return JsonResponse(response_data, status=400)
            elif User.objects.filter(email=email).exists():
                response_data['text'] = "Email already exist."
                return JsonResponse(response_data, status=400)
            else:
                user = User.objects.create_user(username=username, email=email, password=password, first_name=first_name, last_name=last_name)
                user.is_active = False
                user.save()
                
                current_site = get_current_site(request)
                
                
                context = {
                    'user':user, 
                    'domain': current_site.domain,
                    'uid': force_text(urlsafe_base64_encode(force_bytes(user.pk))),
                    'token': account_activation_token.make_token(user),
                }
                
                emailURL = "http://" + str(context.get('domain')) + "/activate/" + str(context.get('uid')) + "/" + str(context.get('token')) + "/"
                emailURL = str(emailURL)
                print(str(emailURL))
                
                
                #send_mail(subject, message, from_email, to_list, fail_silently=True)
                message = render_to_string('acc_active_email.html', {
                    'emailURL': emailURL,
                    'user': user
                })
                mail_subject = "Activate your Check@m account"
                from_email = settings.EMAIL_HOST_USER
                to_list = [email]
                send_mail(mail_subject,message,from_email,to_list,fail_silently=False)
                
                
                # Sending activation link in terminal
                # user.email_user(subject, message)
                # to_email = email
                # email_content = EmailMessage(mail_subject, message, to=[to_email])
                # email_content.send()

                response_data = {}
               
                response_data['emailURL'] = emailURL    
                response_data['name'] = username            
                response_data['title'] = "Success"
                response_data['text'] = 'An email was sent to ' +'<a href="#">'+email+'</a>'+ '. Please confirm your email address to complete the registration'
                
                # render(request, 'acc_active_email.html', {'emailURL': emailURL})
                # auth.login(request, user)
                return JsonResponse(response_data, status=200)
        
        else:
            response_data = {}
            response_data['title'] = 'Password mismatch'
            response_data['text'] = 'Check you passwords again'
            return JsonResponse(response_data, status=400)
   

def activate(request, uidb64, token):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        print("uid: ", uid)
        user = User.objects.get(pk=uid)
        print("user: ", user)
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        auth.login(request,user)
        return redirect('/')
    
    else:
        return HttpResponse('Activation link is invalid!') 








            
def User_login(request):
    response_data = {}
    if request.is_ajax() and request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username, password)
        
        user = auth.authenticate(username=username, password=password)
        
        if user is not None:
            auth.login(request, user)
            response_data = {}
            response_data['text'] = "Signed in successfully"
            return JsonResponse(response_data, status=200)
        
        else:
            response_data = {}
            response_data['text'] = 'Invalid login credentials'
            return JsonResponse(response_data, status=400)
           
    else:
        return redirect('user/login/')
    


def User_logout(request):
    auth.logout(request)
    return redirect('/')
    








# def Register(request):
#     if request.method == 'POST':
#         #Create a form instance and populate it with data from the request
#         form = RegistrationForm(request.POST)
        
#         if form.is_valid():
#             #process the data in form.cleaned_data as required
#             #do any other validation
            
#             return HttpResponseRedirect ('/')
    
#     #if GET(ot any other method) create blank form
#     else:
#         form = RegistrationForm()
        
#     return render(request, 'accounts/test.html', {'form': form})