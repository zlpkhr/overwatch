from django.db import models
from datetime import timedelta


# ---------------------------------------------------------------------------
# Multi-camera support
# ---------------------------------------------------------------------------


class Camera(models.Model):
    """Physical or virtual CCTV camera / RTSP source."""

    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    rtsp_url = models.CharField(max_length=500)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:  # pragma: no cover – simple helper
        return self.name


# ---------------------------------------------------------------------------
# Custom QuerySet helpers for synchronisation (must be defined before Frame)
# ---------------------------------------------------------------------------


class FrameQuerySet(models.QuerySet):
    """Extra helpers for multi-camera synchronisation queries."""

    def synced(self, sync_key: str):
        """Return the *first* frame per camera that matches the given sync_key."""
        frames_by_cam: dict[int, "Frame"] = {}
        for frame in self.filter(sync_key=sync_key).order_by("camera_id", "timestamp"):
            if frame.camera_id not in frames_by_cam:
                frames_by_cam[frame.camera_id] = frame
        return list(frames_by_cam.values())

    def closest_for_all_cameras(self, ts, tolerance_seconds: int = 1):
        """For a given timestamp return closest frames (within tolerance) for every camera.

        Returns a mapping ``{camera_id: frame}``. If some cameras do not have
        frames within the tolerance window they will be missing from the dict.
        """

        if ts is None:
            return {}

        base_ts = ts.replace(microsecond=0)
        # Build candidate sync_keys within ±tolerance_seconds window (inclusive)
        offsets = [0] + [
            d * s
            for d in range(1, tolerance_seconds + 1)
            for s in (-1, 1)
        ]

        result_by_cam: dict[int, "Frame"] = {}

        for offset in offsets:
            candidate_ts = base_ts + timedelta(seconds=offset)
            key = candidate_ts.isoformat()
            for frame in self.synced(key):
                result_by_cam.setdefault(frame.camera_id, frame)
            # Early exit if we already covered all active cameras
            from django.apps import apps  # local import to avoid circular deps

            Camera = apps.get_model("ingest", "Camera")
            if len(result_by_cam) >= Camera.objects.filter(is_active=True).count():
                break

        return result_by_cam


# Manager using the custom queryset


class FrameManager(models.Manager.from_queryset(FrameQuerySet)):  # type: ignore[misc]
    """Default manager for Frame with extra query helpers."""

    # Allows IDEs to understand custom QuerySet methods via type hints.
    _queryset_class = FrameQuerySet


class Frame(models.Model):
    # New: reference to the originating camera (nullable for backward compatibility)
    camera = models.ForeignKey(
        "ingest.Camera",
        related_name="frames",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )

    image = models.ImageField(upload_to="frames")
    timestamp = models.DateTimeField(auto_now_add=True)
    # optional CLIP embedding for the whole frame (semantic representation)
    embedding = models.JSONField(blank=True, null=True)

    # Normalised timestamp (ISO-8601, second resolution) for cross-camera sync
    sync_key = models.CharField(max_length=25, db_index=True, blank=True)

    # Custom manager with helper methods
    objects: "FrameManager" = FrameManager()

    def __str__(self):
        return f"Frame {self.id} at {self.timestamp}"

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def save(self, *args, **kwargs):  # noqa: D401
        """Populate *sync_key* on first save if missing."""
        # Need timestamp to generate sync_key – ensure we have primary key first.
        created = self.pk is None
        super().save(*args, **kwargs)

        if created and not self.sync_key and self.timestamp:
            ts = self.timestamp.replace(microsecond=0)
            self.sync_key = ts.isoformat()
            # avoid infinite recursion by updating only this field
            super().save(update_fields=["sync_key"])


# ------------------------------------------------------------------
# Detected objects metadata
# ------------------------------------------------------------------


class DetectedObject(models.Model):
    """Detected object or OCR text within a frame."""

    frame = models.ForeignKey(
        Frame, related_name="detections", on_delete=models.CASCADE
    )
    label = models.CharField(max_length=100)
    confidence = models.FloatField()

    # Normalised bounding box coordinates (0-1 range)
    x1 = models.FloatField()
    y1 = models.FloatField()
    x2 = models.FloatField()
    y2 = models.FloatField()

    # OCR text (optional)
    text = models.TextField(blank=True, null=True)

    # Embedding vector (list of floats) stored as JSON
    embedding = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [models.Index(fields=["label"])]

    def __str__(self):
        return f"{self.label} ({self.confidence:.2f}) on Frame {self.frame_id}"
