
# Create your models here.
# myapp/models.py

from django.db import models

class TextInput(models.Model):
    text = models.TextField()
