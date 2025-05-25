from celery import shared_task

from ingest.ai import embed_frame, embed_frames_batch
from ingest.models import Frame
from search.chroma_service import ChromaService


@shared_task
def generate_embeddings(frame_id: int):
    frame = Frame.objects.get(id=frame_id)
    embeddings = embed_frame(frame.image.read())
    collection = ChromaService.get_collection()
    collection.upsert(
        embeddings=[embeddings],
        ids=[str(frame.id)],
        metadatas=[{"timestamp": str(frame.timestamp)}],
    )


@shared_task
def generate_embeddings_batch(frame_ids: list[int]):
    frames = list(Frame.objects.filter(id__in=frame_ids))
    images_bytes = [frame.image.read() for frame in frames]
    embeddings_list = embed_frames_batch(images_bytes)
    collection = ChromaService.get_collection()
    for frame, embedding in zip(frames, embeddings_list):
        collection.upsert(
            embeddings=[embedding],
            ids=[str(frame.id)],
            metadatas=[{"timestamp": str(frame.timestamp)}],
        )
