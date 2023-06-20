from django.shortcuts import render

from django.http import HttpResponse
from django import forms


class NameForm(forms.Form):
    name = forms.CharField(label='名字')


def my_view_get(request):
    if request.method == 'GET':
        form = NameForm(request.GET)
        if form.is_valid():
            name = form.cleaned_data['name']
            return HttpResponse(f'Hello, {name}!')
    else:
        form = NameForm()


def my_view_post(request):
    if request.method == 'POST':
        form = NameForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            return HttpResponse(f'Hello, {name}!')
    else:
        form = NameForm()


def home(request):
    return render(request, "home/index.html")


def ajax_get(request):
    return render(request, "home/ajax_get.html")


def success_page(request):
    print("*" * 10)
    return HttpResponse("<h1>Hey this is a Success page</h1>")


def get_test(request):
    return render(request, "home/name_get.html")


def post_test(request):
    return render(request, "home/post_test.html")

