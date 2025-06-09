from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.


def live_stream(request):
    """Render the Live Stream page."""
    return render(request, "stream/live.html")


def get_hls_url(request):
    """Return absolute URL for the HLS master playlist as JSON.

    Optional `slug` query-param allows choosing a different output directory.
    Default: 'live'.
    """
    slug = request.GET.get("slug", "live")
    rel_path = f"{settings.MEDIA_URL}hls/{slug}/index.m3u8"
    abs_url = request.build_absolute_uri(rel_path)
    return JsonResponse({"hls_url": abs_url})
