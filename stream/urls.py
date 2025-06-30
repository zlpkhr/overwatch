from django.urls import path

from stream.views import (
    frame_player,
    frame_sequence,
    get_hls_url,
    latest_frame,
    live_stream,
    live_mosaic,
    multi_frame_player,
    frame_sequence_sync,
)

urlpatterns = [
    path("live/", live_stream, name="live_stream"),
    path("get_hls_url/", get_hls_url, name="get_hls_url"),
    path("frames/play/", frame_player, name="frame_player"),
    path("frames/play/multi/", multi_frame_player, name="multi_frame_player"),
    path("frames/sequence/", frame_sequence, name="frame_sequence"),
    path("frames/sequence/sync/", frame_sequence_sync, name="frame_sequence_sync"),
    path("frames/latest/", latest_frame, name="latest_frame"),
    path("live/mosaic/", live_mosaic, name="live_mosaic"),
]
