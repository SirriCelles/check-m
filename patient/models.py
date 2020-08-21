from django.db import models
from doctor.models import Doctor

# Create your models here.
class Patient(models.Model):
    gender_choices = (
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other')
    )
    
    
    name = models.CharField(max_length=50)
    contact = models.IntegerField()
    email = models.EmailField()
    dob = models.DateField()
    gender = models.CharField(max_length=50, choices=gender_choices)
    location = models.CharField(max_length=50)
    
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)