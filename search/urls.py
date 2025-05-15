from django.urls import path

from search.views import search_frames

urlpatterns = [
    path("search/", search_frames, name="search_frames"),
]
