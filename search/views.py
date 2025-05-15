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

    frames = Frame.objects.exclude(embeddings=None).order_by("-timestamp")
    frame_embeddings = [
        {
            "key": f.id,
            "embeddings": f.embeddings,
        }
        for f in frames
    ]

    documents = compute_similarities(query, frame_embeddings)

    results = []
    for doc in documents[:n_results]:
        frame = frames.get(id=doc["key"])

        results.append(
            {
                "id": frame.id,
                "image": frame.image.url,
                "score": doc["score"],
            }
        )

    return JsonResponse({"results": results})
