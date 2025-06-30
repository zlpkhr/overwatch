from datetime import datetime

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from ingest.models import Frame
from ingest.models import Camera

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


# ---------------------------------------------------------------------------
# Frame-based playback (archive) – lightweight slideshow at 1 FPS
# ---------------------------------------------------------------------------


def frame_player(request):
    """Render the frame-player template.

    Expects a `ts` (ISO-8601) query param indicating the starting point.
    """

    return render(request, "stream/frame_player.html")


def frame_sequence(request):
    """Return a sequence of frame metadata (JSON) starting at timestamp.

    Query params:
        after  ISO-8601 timestamp (required) – return frames > this ts (inclusive on first call)
        count  int (default 60)              – number of frames to return
    """

    count = int(request.GET.get("count", 60))

    before_str = request.GET.get("before")
    after_str = request.GET.get("after")

    if before_str:
        try:
            before_dt = datetime.fromisoformat(before_str)
        except ValueError:
            return JsonResponse({"error": "invalid before format"}, status=400)

        qs = Frame.objects.filter(timestamp__lt=before_dt).order_by("-timestamp")[
            :count
        ]
        qs = list(qs)[::-1]  # chronological order
    elif after_str:
        try:
            after_dt = datetime.fromisoformat(after_str)
        except ValueError:
            return JsonResponse({"error": "invalid after format"}, status=400)

        inclusive = request.GET.get("inc") == "1"
        filter_kwargs = {"timestamp__gte" if inclusive else "timestamp__gt": after_dt}
        qs = Frame.objects.filter(**filter_kwargs).order_by("timestamp")[:count]
    else:
        return JsonResponse({"error": "missing before/after parameter"}, status=400)

    results = []
    for f in qs:
        try:
            img_url = request.build_absolute_uri(f.image.url)
        except Exception:
            img_url = None
        results.append(
            {
                "id": f.id,
                "timestamp": f.timestamp.isoformat(),
                "image_url": img_url,
            }
        )

    # cursors
    next_after = results[-1]["timestamp"] if results else after_str
    prev_before = results[0]["timestamp"] if results else before_str

    return JsonResponse(
        {"results": results, "next_after": next_after, "prev_before": prev_before}
    )


# ---------------------------------------------------------------------------
# Latest frame helper
# ---------------------------------------------------------------------------


@require_GET
def latest_frame(request):
    frame = Frame.objects.order_by("-timestamp").first()
    if not frame:
        return JsonResponse({"error": "no frames"}, status=404)
    try:
        img_url = request.build_absolute_uri(frame.image.url)
    except Exception:
        img_url = None
    return JsonResponse(
        {"id": frame.id, "timestamp": frame.timestamp.isoformat(), "image_url": img_url}
    )


# ---------------------------------------------------------------------------
# NEW: Live mosaic (multi-camera wall)
# ---------------------------------------------------------------------------


def live_mosaic(request):
    """Render a CCTV-room style mosaic of all active camera streams."""

    cameras = Camera.objects.filter(is_active=True).order_by("name")
    return render(request, "stream/mosaic.html", {"cameras": cameras})


# ---------------------------------------------------------------------------
# Multi-camera frame player (synchronised by sync_key)
# ---------------------------------------------------------------------------


def multi_frame_player(request):
    """Renders the multi-camera frame player starting at given timestamp."""

    cams = list(Camera.objects.filter(is_active=True).order_by("name"))
    import json

    cameras_json = json.dumps([c.slug for c in cams])
    return render(
        request,
        "stream/multi_frame_player.html",
        {"cameras_json": cameras_json, "cameras": cams},
    )


@require_GET
def frame_sequence_sync(request):
    """Return sequence of synchronised frames across cameras (JSON).

    Query params:
        after   ISO-8601 timestamp – start (inclusive)
        count   int – number of seconds to return (default 60)
    """

    count = int(request.GET.get("count", 60))
    after_str = request.GET.get("after")

    if not after_str:
        return JsonResponse({"error": "missing after parameter"}, status=400)

    try:
        after_dt = datetime.fromisoformat(after_str)
    except ValueError:
        return JsonResponse({"error": "invalid after format"}, status=400)

    # Get distinct sync_keys >= after_dt, limit to count
    sync_keys_qs = (
        Frame.objects.filter(timestamp__gte=after_dt)
        .order_by("sync_key")
        .values_list("sync_key", flat=True)
        .distinct()
    )[:count]

    cameras = list(Camera.objects.filter(is_active=True))
    slug_map = {c.id: c.slug for c in cameras}

    results = []
    for key in sync_keys_qs:
        frame_map = {c.slug: None for c in cameras}
        frames = Frame.objects.filter(sync_key=key)
        for f in frames:
            slug = slug_map.get(f.camera_id)
            if not slug:
                continue
            try:
                img_url = request.build_absolute_uri(f.image.url)
            except Exception:
                img_url = None
            frame_map[slug] = {
                "id": f.id,
                "timestamp": f.timestamp.isoformat(),
                "image_url": img_url,
            }
        results.append({"sync_key": key, "frames": frame_map})

    next_after = results[-1]["sync_key"] if results else after_str

    return JsonResponse({"results": results, "next_after": next_after})
