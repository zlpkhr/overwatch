from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render


def live_stream(request):
    """Render the Live Stream page (overwatch/live.html)."""
    return render(request, "overwatch/live.html")


def get_hls_url(request):
    """Return the absolute URL of the HLS playlist (JSON).

    A `slug` GET parameter can be provided to choose a different output
    directory, but for single-tenant setups the default (`live`) is fine.
    """

    slug = request.GET.get("slug", "live")
    rel_path = f"{settings.MEDIA_URL}hls/{slug}/index.m3u8"
    abs_url = request.build_absolute_uri(rel_path)
    return JsonResponse({"hls_url": abs_url}) 