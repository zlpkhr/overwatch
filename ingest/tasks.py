from celery import shared_task
from ingest.models import Frame


@shared_task
def generate_embeddings(frame_id: int):
    frame = Frame.objects.get(id=frame_id)
    frame.embeddings = generate_embeddings(frame.image)
    frame.save()