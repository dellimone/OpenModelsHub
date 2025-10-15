from django.urls import path
from . import views

app_name = 'frontend'

urlpatterns = [
    # Home
    path('', views.home, name='home'),

    # Models
    path('models/', views.model_list, name='model_list'),
    path('models/<uuid:model_id>/', views.model_detail, name='model_detail'),

    # Datasets
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/<uuid:dataset_id>/', views.dataset_detail, name='dataset_detail'),

    # Experiments
    path('experiments/', views.experiment_list, name='experiment_list'),
    path('experiments/<uuid:experiment_id>/', views.experiment_detail, name='experiment_detail'),

    # Researchers
    path('researchers/', views.researcher_list, name='researcher_list'),
    path('researchers/<uuid:researcher_id>/', views.researcher_detail, name='researcher_detail'),

    # Organizations
    path('organizations/', views.organization_list, name='organization_list'),

    # Resources
    path('resources/', views.resources_dashboard, name='resources_dashboard'),
    path('resources/<uuid:resource_id>/', views.resource_detail, name='resource_detail'),

    # Environmental
    path('environmental/', views.environmental_dashboard, name='environmental_dashboard'),
    path('sustainability/', views.sustainability_report, name='sustainability_report'),

    # Search
    path('search/', views.search, name='search'),
]
