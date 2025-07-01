from django.urls import path
from ingest.views import CameraListView, CameraCreateView, CameraUpdateView, CameraDeleteView, cameras_json

urlpatterns = [
    path("cameras/", CameraListView.as_view(), name="camera_list"),
    path("cameras/add/", CameraCreateView.as_view(), name="camera_add"),
    path("cameras/<int:pk>/edit/", CameraUpdateView.as_view(), name="camera_edit"),
    path("cameras/<int:pk>/del/", CameraDeleteView.as_view(), name="camera_delete"),

    # API
    path("api/cameras/", cameras_json, name="api_cameras"),
] 