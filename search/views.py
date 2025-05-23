# Create your views here.

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ingest.models import Frame
from search.ai import compute_similarities


@require_GET
def search_frames(request):
    query = request.GET.get("q")
    n_results = int(request.GET.get("n", 10))

    if not query:
        return JsonResponse({"error": "Missing query parameter q"}, status=400)

    frames = Frame.objects.exclude(embeddings=[]).order_by("-timestamp")

    frame_embeddings = [
        {
            "key": f.id,
            "embeddings": f.embeddings,
        }
        for f in frames
    ]

    documents = compute_similarities(query, frame_embeddings)
    top_score = documents[0]["score"] or 0
    bottom_score = documents[-1]["score"] or 0
    score_range = top_score - bottom_score

    results = []
    for doc in documents[:n_results]:
        frame = frames.get(id=doc["key"])

        results.append(
            {
                "id": frame.id,
                "image_url": request.build_absolute_uri(frame.image.url),
                "score": (doc["score"] - bottom_score) / score_range,
                "cosine_similarity": doc["score"],
            }
        )

    return JsonResponse({"results": results})
