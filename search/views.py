# Create your views here.

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ingest.models import Frame
from search.ai import embed_query
from search.faiss_service import FaissIndexService


@require_GET
def search_frames(request):
    query = request.GET.get("q")
    n_results = int(request.GET.get("n", 10))

    if not query:
        return JsonResponse({"error": "Missing query parameter q"}, status=400)

    query_embedding = embed_query(query)
    faiss_service = FaissIndexService.get_instance()
    results = faiss_service.search(query_embedding, n_results)

    if not results:
        return JsonResponse({"results": []})

    # Normalize scores (distances) to [0, 1] range (lower is better for L2)
    dists = [score for _, score in results]
    min_dist, max_dist = min(dists), max(dists)
    range_dist = max_dist - min_dist if max_dist != min_dist else 1.0

    response = []
    for frame_id, dist in results:
        try:
            frame = Frame.objects.get(id=frame_id)
            response.append({
                "id": frame.id,
                "image_url": request.build_absolute_uri(frame.image.url),
                "score": 1.0 - ((dist - min_dist) / range_dist),  # higher is better
                "faiss_distance": dist,
            })
        except Frame.DoesNotExist:
            continue

    return JsonResponse({"results": response})
