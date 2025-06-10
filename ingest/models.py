from django.db import models


class Frame(models.Model):
    image = models.ImageField(upload_to="frames")
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Frame {self.id} at {self.timestamp}"


# ------------------------------------------------------------------
# Detected objects metadata
# ------------------------------------------------------------------

class DetectedObject(models.Model):
    """Detected object or OCR text within a frame."""

    frame = models.ForeignKey(Frame, related_name="detections", on_delete=models.CASCADE)
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
