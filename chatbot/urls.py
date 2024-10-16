from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot, name='home'),
    path('chatbot/', views.chatbot, name='chatbot'),
]