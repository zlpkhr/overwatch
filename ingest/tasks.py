from celery import shared_task

from ingest.ai import embed_frame, embed_frames_batch
from ingest.models import Frame


@shared_task
def generate_embeddings(frame_id: int):
    frame = Frame.objects.get(id=frame_id)

    embeddings = embed_frame(frame.image.read())

    frame.embeddings = embeddings
    frame.save()


@shared_task
def generate_embeddings_batch(frame_ids: list[int]):
    frames = list(Frame.objects.filter(id__in=frame_ids))
    images_bytes = [frame.image.read() for frame in frames]
    embeddings_list = embed_frames_batch(images_bytes)
    for frame, embedding in zip(frames, embeddings_list):
        frame.embeddings = embedding
        frame.save()
