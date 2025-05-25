# Create your views here.

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ingest.models import Frame
from search.ai import embed_query
from search.chroma_service import ChromaService


@require_GET
def search_frames(request):
    query = request.GET.get("q")
    n_results = int(request.GET.get("n", 10))

    if not query:
        return JsonResponse({"error": "Missing query parameter q"}, status=400)

    # Use our own CLIP embed_query to ensure correct dimension
    query_embedding = embed_query(query)
    collection = ChromaService.get_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    if not ids:
        return JsonResponse({"results": []})
    min_dist, max_dist = min(distances), max(distances)
    range_dist = max_dist - min_dist if max_dist != min_dist else 1.0
    response = []
    for frame_id, dist in zip(ids, distances):
        try:
            frame = Frame.objects.get(id=frame_id)
            response.append(
                {
                    "id": frame.id,
                    "image_url": request.build_absolute_uri(frame.image.url),
                    "score": 1.0 - ((dist - min_dist) / range_dist),
                    "chroma_distance": dist,
                }
            )
        except Frame.DoesNotExist:
            continue
    return JsonResponse({"results": response})
