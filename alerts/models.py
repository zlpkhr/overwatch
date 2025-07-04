from django.db import models


class AlertRule(models.Model):
    """Rule that triggers an alert when an object detection matches."""

    name = models.CharField(max_length=120)
    active = models.BooleanField(default=True)

    # Basic metadata conditions ------------------------------------------------
    label = models.CharField(
        max_length=100, blank=True, help_text="Match specific object label (optional)"
    )
    min_confidence = models.FloatField(
        default=0.0, help_text="Minimum confidence threshold (0-1)"
    )

    # OCR text condition --------------------------------------------------------
    text_contains = models.CharField(
        max_length=200,
        blank=True,
        help_text="Case-insensitive substring search on OCR text",
    )

    # User friendly description field -----------------------------------------
    description = models.CharField(
        max_length=200,
        blank=True,
        help_text="Natural language description for the alert rule (auto embedded)",
    )

    # Embedding similarity condition -------------------------------------------
    embedding = models.JSONField(
        blank=True,
        null=True,
        help_text="Reference embedding vector (list of floats) for cosine similarity",
    )

    # similarity threshold expressed as percentage (0-100)
    min_similarity = models.PositiveSmallIntegerField(
        default=15,
        help_text="Minimum CLIP similarity score (0-100) required to trigger this rule",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        # Auto-generate embedding from description text if provided
        if self.description:
            try:
                from ingest.ai import embed_text  # local import to avoid circular deps

                self.embedding = embed_text(self.description)
            except Exception:
                pass  # if embedding fails, continue saving
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name


class Alert(models.Model):
    """An alert generated by a rule for a specific frame/detection."""

    rule = models.ForeignKey(AlertRule, on_delete=models.CASCADE)
    frame = models.ForeignKey("ingest.Frame", on_delete=models.CASCADE)
    detection = models.ForeignKey(
        "ingest.DetectedObject", null=True, on_delete=models.SET_NULL
    )

    timestamp = models.DateTimeField(auto_now_add=True)
    acknowledged = models.BooleanField(default=False)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"Alert {self.id} – {self.rule.name}"


class AlertReferenceImage(models.Model):
    """User-provided reference image used for similarity-based alert rules."""

    rule = models.ForeignKey(
        AlertRule, related_name="reference_images", on_delete=models.CASCADE
    )
    image = models.ImageField(upload_to="alert_refs")
    # CLIP embedding vector of the image
    embedding = models.JSONField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        # Compute and store embedding *before* the file is permanently saved.
        # Important: do not close the underlying UploadedFile – Django still
        # needs to read it when persisting to storage.
        if self.image and (self._state.adding or not self.embedding):
            try:
                from ingest.ai import embed_frame  # local import to avoid circular deps

                # Ensure pointer at start
                if hasattr(self.image, "seek"):
                    self.image.seek(0)
                bytes_data = self.image.read()
                # Rewind so subsequent save can still read
                if hasattr(self.image, "seek"):
                    self.image.seek(0)

                self.embedding = embed_frame(bytes_data)
            except Exception:
                pass  # don't block save if embedding fails
        super().save(*args, **kwargs)

    def __str__(self):
        return f"RefImage {self.id} for {self.rule.name}"
