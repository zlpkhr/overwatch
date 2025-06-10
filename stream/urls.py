from django.urls import path

from stream.views import frame_player, frame_sequence, get_hls_url, live_stream

urlpatterns = [
    path("live/", live_stream, name="live_stream"),
    path("get_hls_url/", get_hls_url, name="get_hls_url"),
    path("frames/play/", frame_player, name="frame_player"),
    path("frames/sequence/", frame_sequence, name="frame_sequence"),
]
