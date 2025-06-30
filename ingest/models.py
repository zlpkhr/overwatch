from django.db import models


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
