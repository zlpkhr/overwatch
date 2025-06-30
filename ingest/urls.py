from django.urls import path

from ingest.views import (
    CameraListView,
    CameraCreateView,
    CameraUpdateView,
    CameraDeleteView,
)

urlpatterns = [
    path("streams/", CameraListView.as_view(), name="camera_list"),
    path("streams/add/", CameraCreateView.as_view(), name="camera_add"),
    path("streams/<slug:slug>/edit/", CameraUpdateView.as_view(), name="camera_edit"),
    path("streams/<slug:slug>/delete/", CameraDeleteView.as_view(), name="camera_delete"),
] 