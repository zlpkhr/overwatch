from django.urls import path

from stream.views import live_stream, get_hls_url

urlpatterns = [
    path("live/", live_stream, name="live_stream"),
    path("get_hls_url/", get_hls_url, name="get_hls_url"),
] 