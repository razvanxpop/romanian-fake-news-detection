from django.urls import path
from . import views

app_name = 'fake_news_ui'

urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),

    path('classify/', views.IndexView.as_view(), name='classify_article'),

    path('word-similarity/', views.WordSimilarityView.as_view(), name='word_similarity'),
]
