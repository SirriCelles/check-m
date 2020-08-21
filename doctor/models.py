from django.db import models


# Create your models here.
class Doctor(models.Model):
    
    name = models.CharField(max_length=50)
    contact = models.IntegerField()
    email = models.EmailField()
    service_sector = models.CharField(max_length=50)
    address = models.CharField(max_length=50)
    app_status = models.BooleanField()