from django.urls import path, re_path

from . import views

urlpatterns = [
    re_path(r'^symptom-checker/$', views.Show_symptom_page, name='symptoms_view'),
    re_path(r'^symptom-checker/diagnosis/$', views.getPrediction, name='get_diagnosis'),
    path('symptom-checker/symptom/', views.specific_symptom, name="specific_symptom"),
    # re_path('symptom-checker/test/', views.testModel, name="test_model"),
    path('symptom-checker/symptom/#step-3/', views.getNumOfDays, name="num_of_days"),
    path('symptom-checker/tree_travesal/', views.tree_to_code, name="tree_to_code")
]