from celery import shared_task

from ingest.ai import embed_frame
from ingest.models import Frame


@shared_task
def generate_embeddings(frame_id: int):
    frame = Frame.objects.get(id=frame_id)
    embeddings = embed_frame(frame.image.read())
    print(embeddings)
