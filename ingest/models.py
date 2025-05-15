from django.db import models


class Frame(models.Model):
    image = models.ImageField(upload_to="frames")
    timestamp = models.DateTimeField(auto_now_add=True)
    embeddings = models.JSONField(default=list)

    def __str__(self):
        return f"Frame {self.id} at {self.timestamp}"
