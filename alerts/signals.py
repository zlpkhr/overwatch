from django.db.models.signals import post_save
from django.dispatch import receiver

from alerts.engine import evaluate_object
from ingest.models import DetectedObject


@receiver(post_save, sender=DetectedObject)
def detected_object_post_save(
    sender, instance: DetectedObject, created: bool, **kwargs
):
    """Run alert evaluation whenever a DetectedObject is created."""
    if created:
        evaluate_object(instance)
