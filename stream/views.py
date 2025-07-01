from datetime import datetime, timedelta

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from ingest.models import Frame, Camera

# Create your views here.


def live_stream(request):
    """Render the Live Stream page."""
    return render(request, "stream/live.html")


# ---------------------------------------------------------------------------
# HLS playlist helpers
# ---------------------------------------------------------------------------


def get_hls_url(request):
    """Return HLS URL(s) for one or all cameras as JSON.

    Query params:
        camera  – (int) camera id. If omitted, returns mapping of every camera.
    """

    camera_id = request.GET.get("camera")

    if camera_id:
        try:
            cam = Camera.objects.get(id=int(camera_id))
        except (Camera.DoesNotExist, ValueError):
            return JsonResponse({"error": "camera not found"}, status=404)
        rel_path = f"{settings.MEDIA_URL}hls/{cam.id}/index.m3u8"
        abs_url = request.build_absolute_uri(rel_path)
        return JsonResponse({"camera_id": cam.id, "hls_url": abs_url})

    # else: all cameras
    cams = Camera.objects.all()
    data = {}
    for cam in cams:
        rel_path = f"{settings.MEDIA_URL}hls/{cam.id}/index.m3u8"
        data[str(cam.id)] = request.build_absolute_uri(rel_path)
    return JsonResponse({"hls_urls": data})


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
# Sync frames across cameras at given timestamp
# ---------------------------------------------------------------------------


@require_GET
def sync_frames(request):
    """Return nearest frame for each camera at a given timestamp.

    Query params:
        ts ISO-8601 timestamp (required)
        tol float seconds tolerance (default 1.0)
    """

    ts_str = request.GET.get("ts")
    if not ts_str:
        return JsonResponse({"error": "missing ts"}, status=400)
    try:
        target_ts = datetime.fromisoformat(ts_str)
    except ValueError:
        return JsonResponse({"error": "invalid ts"}, status=400)

    tol = float(request.GET.get("tol", 1.0))

    cams = Camera.objects.all()
    data = {}
    for cam in cams:
        frames_qs = Frame.objects.filter(
            camera=cam,
            timestamp__gte=target_ts - timedelta(seconds=tol),
            timestamp__lte=target_ts + timedelta(seconds=tol),
        )
        if not frames_qs.exists():
            continue
        # choose the frame whose timestamp is closest to target_ts
        frame = min(frames_qs, key=lambda f: abs((f.timestamp - target_ts).total_seconds()))
        if frame:
            try:
                img_url = request.build_absolute_uri(frame.image.url)
            except Exception:
                img_url = None
            data[str(cam.id)] = {
                "id": frame.id,
                "timestamp": frame.timestamp.isoformat(),
                "image_url": img_url,
            }
    return JsonResponse({"timestamp": target_ts.isoformat(), "frames": data})
