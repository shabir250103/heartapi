from django.urls import path
from .views import PredictHeartSound

urlpatterns = [
    path('predict/', PredictHeartSound.as_view(), name='predict'),
]
