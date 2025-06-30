# Create your views here.

from django.urls import reverse_lazy
from django.views import generic

from ingest.models import Camera


class CameraListView(generic.ListView):
    model = Camera
    template_name = "ingest/camera_list.html"
    context_object_name = "cameras"


class CameraCreateView(generic.CreateView):
    model = Camera
    fields = ["name", "slug", "rtsp_url", "is_active"]
    template_name = "ingest/camera_form.html"
    success_url = reverse_lazy("camera_list")


class CameraUpdateView(generic.UpdateView):
    model = Camera
    fields = ["name", "rtsp_url", "is_active"]
    slug_field = "slug"
    slug_url_kwarg = "slug"
    template_name = "ingest/camera_form.html"
    success_url = reverse_lazy("camera_list")


class CameraDeleteView(generic.DeleteView):
    model = Camera
    slug_field = "slug"
    slug_url_kwarg = "slug"
    template_name = "ingest/camera_confirm_delete.html"
    success_url = reverse_lazy("camera_list")
