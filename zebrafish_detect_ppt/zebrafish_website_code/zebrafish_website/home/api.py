from django.shortcuts import render, redirect

from django.http import HttpResponse
from django import forms


from home.test.test import main
from django.http import JsonResponse
from django.shortcuts import render
from home.models import IMG


def ImagePost(request):
     return render(request,'home/image_post.html')
    

def ImgUpload(request):
    file_img = request.FILES['img']  # 獲取文件對象
    
    image_name = main.m(file_img)#斑馬魚辨識
    image_detect_path = f"/media/detect/{image_name}.jpg"
    csv_path = f"/media/csv_file/{image_name}.csv"
    


    
    file_img_name = file_img.name
    image_path = ("/media/upload/"+ file_img_name)
    # print(csv_path)
    # print(image_detect_path)
    response = { 
        'url1' : image_path,
        'url2' : image_detect_path,
        'url3' : csv_path

    }
    

    try:

        #return JsonResponse(1, safe=False)
        return JsonResponse(response, safe=False)
    except Exception as e:
        print(e)
        return JsonResponse(0, safe=False)
    

    