from django.urls import path

from . import views

urlpatterns = [
   path('', views.search_engine, name='search_engine'),

   
]
