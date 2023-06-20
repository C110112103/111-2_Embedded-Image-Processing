"""core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from home import views
from home import api

from django.conf import settings
from django.conf.urls.static import static

 
urlpatterns = [
    
    path('ajax_get/', views.ajax_get, name="ajax_get"),

    path('home', views.home, name="home"),

    path('success_page/', views.success_page, name="success_page"),

    path('get_test/', views.get_test, name="get_test"),

    path('post_test/', views.post_test, name="post_test"),

    path('my_view_get/', views.my_view_get, name="my_view_get"),

    path('my_view_post/', views.my_view_post, name="my_view_post"),

    path('admin/', admin.site.urls),

    path('', api.ImagePost, name="ImagePost"),

    path('api/upload_img/', api.ImgUpload, name="ImgUpload"),

    
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
