from django.db import models


class Frame(models.Model):
    image = models.ImageField(upload_to="frames")
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Frame {self.id} at {self.timestamp}"


class Camera(models.Model):
    """Represents an IP / RTSP camera that can be restreamed as HLS."""

    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    rtsp_url = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def hls_relative_path(self) -> str:
        """Return the MEDIA-relative path to the master HLS playlist."""
        return f"hls/{self.slug}/index.m3u8"

    def hls_url(self, request=None) -> str:
        """Return an absolute URL that clients can use to play the live stream.

        If a request is supplied we build an absolute URI with the request host,
        otherwise we return a MEDIA_URL-relative path that can be manually
        combined on the client.
        """
        from django.conf import settings

        rel = f"{settings.MEDIA_URL}{self.hls_relative_path()}"
        if request is not None:
            return request.build_absolute_uri(rel)
        return rel
