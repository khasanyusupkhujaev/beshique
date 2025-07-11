from django.urls import path
from .views import home, faq, blogs

urlpatterns = [
    path('', home, name='home'),
    path('faq/', faq, name='faq'),
    path('blogs/', blogs, name='blog')
]