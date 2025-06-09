from django.urls import path

from stream.views import get_hls_url, live_stream

urlpatterns = [
    path("live/", live_stream, name="live_stream"),
    path("get_hls_url/", get_hls_url, name="get_hls_url"),
]
