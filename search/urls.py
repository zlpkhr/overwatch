from django.urls import path

from search.views import search_frames, search_page, search_timestamps, frame_detections

urlpatterns = [
    path("search/", search_page, name="search_page"),
    path("search/search_frames/", search_frames, name="search_frames"),
    path("search/search_timestamps/", search_timestamps, name="search_timestamps"),
    path("search/frame/<int:frame_id>/detections/", frame_detections, name="frame_detections"),
]
