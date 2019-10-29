from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .image_detect import gun_detect
from django.contrib.auth.forms import UserCreationForm
import numpy as np
import cv2
# Create your views here.
def homepage(request):   
    if request.method =="POST":
        upload_file = request.FILES['myfile']
        fs = FileSystemStorage()
        fs.save(upload_file.name, upload_file)
        temp = gun_detect(upload_file.name)
        result = temp.save()
        return render(request, 'main/home.html', {'result': result})
    return render(request, template_name= "main/home.html")
