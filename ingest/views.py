# ---------------------------------------------------------------------------
# Camera CRUD (server-rendered)
# ---------------------------------------------------------------------------


from django.forms import ModelForm
from django.http import JsonResponse
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView, DeleteView, ListView, UpdateView

from ingest.models import Camera


class CameraForm(ModelForm):
    class Meta:
        model = Camera
        fields = ["rtsp_url"]


class CameraListView(ListView):
    model = Camera
    template_name = "ingest/camera_list.html"


class CameraCreateView(CreateView):
    model = Camera
    form_class = CameraForm
    template_name = "ingest/camera_form.html"
    success_url = reverse_lazy("camera_list")


class CameraUpdateView(UpdateView):
    model = Camera
    form_class = CameraForm
    template_name = "ingest/camera_form.html"
    success_url = reverse_lazy("camera_list")


class CameraDeleteView(DeleteView):
    model = Camera
    template_name = "ingest/camera_confirm_delete.html"
    success_url = reverse_lazy("camera_list")


# ---------------------------------------------------------------------------
# JSON API – list cameras
# ---------------------------------------------------------------------------


def cameras_json(request):
    data = [
        {"id": cam.id, "rtsp_url": cam.rtsp_url, "created_at": cam.created_at.isoformat()}
        for cam in Camera.objects.all().order_by("id")
    ]
    return JsonResponse({"cameras": data})
