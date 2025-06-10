# Create your views here.

import base64
import json

import openai
import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from ingest.models import Frame
from search.ai import embed_query
from search.chroma_service import ChromaService

# Prompt for LLM-based query expansion
prompt = """
Expand a user query for a CCTV footage analysis and search tool to generate multiple variations tailored to finding relevant frames. The number of expanded queries should match the user's request.

Steps:
1. Understand the user query and what they are searching for in the CCTV footage.
2. Generate multiple variations of the query, focusing on different aspects that might appear in the footage (such as angles, times, related objects, or actions).
3. Ensure the expanded queries maintain the original search intent and help retrieve all relevant frames.
4. The number of expanded queries should match the user-specified amount (n).

Output format:
Return the expanded queries as json object with a single key "queries" and a list of natural language strings as value.

Example:
Input: 
Expanded queries amount: 3
Query: "person wearing a red hoodie entering from the left"

Output:
{
  "queries": [
    "individual in red hoodie coming through the left entrance",
    "someone with a red sweatshirt approaching from the left side",
    "person in red hoodie seen on the left side entry point"
  ]
}

Notes:
- Each expanded query should capture potential variations in phrasing and elements associated with the original query.
- Use different vocabulary and phrasing styles to cover a range of possible expressions.
"""


def expand_query_llm(query, n_expansions=3):
    """
    Use OpenAI LLM to expand a user query into multiple natural language variations.
    This helps cover different phrasings and aspects for more robust search.
    Returns a list of expanded queries (strings).
    """
    try:
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.responses.create(
            model="gpt-4.1",
            instructions=prompt,
            input=f"Expanded queries amount: {n_expansions}\nQuery: {query}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "queries_list",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "description": "A list of query strings.",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["queries"],
                        "additionalProperties": False,
                    },
                }
            },
            reasoning={},
            tools=[],
            temperature=1,
            max_output_tokens=2048,
            top_p=1,
            store=True,
        )
        res = json.loads(response.output_text)["queries"]
        print(res)
        return res
    except Exception as e:
        print(f"expand_query_llm error: {e}")
        return [query]


def encode_image_to_base64(image_field):
    """
    Read a Django ImageField and encode its contents as a base64 string.
    Used for sending images to the Jina reranker API.
    """
    image_field.open("rb")
    encoded = base64.b64encode(image_field.read()).decode("utf-8")
    image_field.close()
    return encoded


def calculate_score(dist, min_dist, max_dist):
    """
    Normalize a distance value to a [0,1] score (higher is better).
    If all distances are equal, returns 1.0 for all.
    """
    dist = float(dist)
    min_dist = float(min_dist)
    max_dist = float(max_dist)
    if max_dist == min_dist:
        return 1.0
    return 1.0 - ((dist - min_dist) / (max_dist - min_dist))


def get_candidate_frames(sorted_results, n_results):
    """
    Given sorted (frame_id, distance) pairs, fetch Frame objects for the top candidates.
    Returns (candidate_ids, id_to_frame) where:
      - candidate_ids: list of string frame IDs in order
      - id_to_frame: dict mapping string frame ID to Frame instance
    """
    candidate_ids = [frame_id for frame_id, _ in sorted_results[:n_results]]
    frames = Frame.objects.filter(id__in=candidate_ids)
    id_to_frame = {str(f.id): f for f in frames}
    return candidate_ids, id_to_frame


def call_jina_reranker(query, images_for_rerank, n_results):
    """
    Call the Jina reranker API (m0 model) with the original query and a list of images (base64).
    Returns a tuple (reranked_indices, rerank_scores):
      - reranked_indices: list of indices (into id_order/images_for_rerank) in reranked order
      - rerank_scores: list of Jina relevance scores in reranked order
    If the API call fails, returns the original order and None for scores.
    """
    reranked_indices = list(range(len(images_for_rerank)))
    rerank_scores = None
    if not images_for_rerank:
        return reranked_indices, rerank_scores
    try:
        # Get your Jina AI API key for free: https://jina.ai/?sui=apikey
        JINA_API_KEY = settings.JINA_API_KEY
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": "jina-reranker-m0",
            "query": query,
            "documents": images_for_rerank,
            "top_n": n_results,
            "return_documents": False,
        }
        resp = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Sort results by Jina's relevance score (descending)
            sorted_results = sorted(
                data["results"], key=lambda x: -x["relevance_score"]
            )
            reranked_indices = [r["index"] for r in sorted_results]
            rerank_scores = [r["relevance_score"] for r in sorted_results]
    except Exception:
        pass  # fallback to original order
    return reranked_indices, rerank_scores


def compose_response(
    reranked_indices,
    id_order,
    id_to_frame,
    best,
    sorted_results,
    request,
    rerank_scores=None,
):
    """
    Compose the final response list in reranked order for the API response.
    - reranked_indices: order of results after reranking (indices into id_order)
    - id_order: original order of candidate frame IDs (as strings)
    - id_to_frame: mapping from frame ID to Frame instance
    - best: dict mapping frame ID to chroma distance
    - sorted_results: original sorted (frame_id, distance) pairs from Chroma
    - request: Django request (for building absolute image URLs)
    - rerank_scores: optional list of Jina reranker scores (same order as reranked_indices)
    Returns a list of dicts, each representing a result.
    """
    # Only use the distances of the returned candidates for normalization
    candidate_dists = [
        float(best.get(id_order[idx], 1.0))
        for idx in reranked_indices
        if idx < len(id_order)
    ]
    # Debug prints for future refactoring/diagnosis
    print(f"DEBUG: best={best}")
    print(f"DEBUG: id_order={id_order}")
    print(f"DEBUG: reranked_indices={reranked_indices}")
    print(f"DEBUG: candidate_dists={candidate_dists}")
    response = []
    if candidate_dists:
        min_dist = min(candidate_dists)
        max_dist = max(candidate_dists)
    else:
        min_dist = max_dist = 1.0
    for i, idx in enumerate(reranked_indices):
        if idx >= len(id_order):
            continue
        frame_id = id_order[idx]
        frame = id_to_frame.get(frame_id)
        if not frame:
            continue
        dist = float(best.get(frame_id, 1.0))
        print(f"DEBUG: frame_id={frame_id}, dist={dist}")
        try:
            image_url = request.build_absolute_uri(frame.image.url)
        except Exception:
            image_url = None
        score = calculate_score(dist, min_dist, max_dist)
        result = {
            "id": frame.id,
            "image_url": image_url,
            "timestamp": frame.timestamp.isoformat(),
            "score": score,  # normalized chroma distance (higher is better)
            "chroma_distance": dist,  # raw chroma distance
        }
        if rerank_scores and i < len(rerank_scores):
            result["rerank_score"] = rerank_scores[i]  # Jina reranker relevance score
        response.append(result)
    return response


@require_GET
def search_frames(request):
    """
    Main search endpoint for CCTV frame retrieval.
    Steps:
      1. Expand the user query using LLM for robust search coverage.
      2. Embed all expanded queries and search Chroma vector DB for candidate frames.
      3. Deduplicate and sort candidates by best (lowest) distance.
      4. Fetch candidate frame images and encode to base64 for reranking.
      5. Call Jina reranker API to rerank candidates by relevance to the query.
      6. Compose and return the final response, including both chroma and reranker scores.
    """
    query = request.GET.get("q")
    n_results = int(request.GET.get("n", 10))

    if not query:
        return JsonResponse({"error": "Missing query parameter q"}, status=400)

    # Step 1: LLM-based multi-query expansion
    try:
        expanded_queries = expand_query_llm(query, n_expansions=3)
    except Exception:
        expanded_queries = [query]

    # Step 2: Embed all expanded queries and search Chroma
    query_embeddings = [embed_query(q) for q in expanded_queries]
    collection = ChromaService.get_collection()
    all_results = []
    for emb in query_embeddings:
        results = collection.query(query_embeddings=[emb], n_results=n_results)
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metas = (
            results.get("metadatas", [[]])[0]
            if results.get("metadatas")
            else [{}] * len(ids)
        )
        for _id, dist, meta in zip(ids, distances, metas):
            frame_id = str(meta.get("frame_id") or _id.split("_", 1)[0])
            all_results.append((frame_id, dist))
    # Step 3: Deduplicate by id, keep best (lowest) distance
    best = {}
    for frame_id, dist in all_results:
        if frame_id not in best or dist < best[frame_id]:
            best[frame_id] = dist
    # Step 4: Sort by distance
    sorted_results = sorted(best.items(), key=lambda x: x[1])
    if not sorted_results:
        return JsonResponse({"results": []})
    # Step 5: Prepare candidate frames and images for reranking
    candidate_ids, id_to_frame = get_candidate_frames(sorted_results, n_results)
    images_for_rerank = []
    id_order = []
    for frame_id in candidate_ids:
        frame = id_to_frame.get(str(frame_id))
        if frame:
            try:
                img_b64 = encode_image_to_base64(frame.image)
                images_for_rerank.append({"image": img_b64})
                id_order.append(str(frame.id))
            except Exception:
                continue
    # Step 6: Call Jina reranker and compose response
    reranked_indices, rerank_scores = call_jina_reranker(
        query, images_for_rerank, n_results
    )
    response = compose_response(
        reranked_indices,
        id_order,
        id_to_frame,
        best,
        sorted_results,
        request,
        rerank_scores,
    )
    return JsonResponse({"results": response})


def search_page(request):
    """
    Render the Bootstrap-based search interface for CCTV frame search.
    """
    return render(request, "search/search.html")


# ---------------------------------------------------------------------------
# New lightweight endpoint: returns a list of {frame_id, timestamp, score}
# for timeline visualisation on the frontend.
# ---------------------------------------------------------------------------


@require_GET
def search_timestamps(request):
    """Similar to search_frames but returns only timestamps for matched frames.

    Query params:
        q   – search query (required)
        n   – number of results (default 100)
    """

    query = request.GET.get("q")
    n_results = int(request.GET.get("n", 100))

    if not query:
        return JsonResponse({"error": "Missing query parameter q"}, status=400)

    # We re-use same search approach without reranking (faster for timeline)
    expanded_queries = [query]
    try:
        expanded_queries = expand_query_llm(query, n_expansions=3)
    except Exception:
        pass

    query_embeddings = [embed_query(q) for q in expanded_queries]
    collection = ChromaService.get_collection()
    all_results = []
    for emb in query_embeddings:
        results = collection.query(query_embeddings=[emb], n_results=n_results)
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metas = (
            results.get("metadatas", [[]])[0]
            if results.get("metadatas")
            else [{}] * len(ids)
        )
        for _id, dist, meta in zip(ids, distances, metas):
            frame_id = str(meta.get("frame_id") or _id.split("_", 1)[0])
            all_results.append((frame_id, dist))

    best = {}
    for frame_id, dist in all_results:
        if frame_id not in best or dist < best[frame_id]:
            best[frame_id] = dist

    sorted_results = sorted(best.items(), key=lambda x: x[1])[:n_results]

    frames = Frame.objects.filter(id__in=[fid for fid, _ in sorted_results])
    id_to_frame = {str(f.id): f for f in frames}

    response_data = []
    for frame_id, dist in sorted_results:
        frame = id_to_frame.get(str(frame_id))
        if not frame:
            continue
        try:
            image_url = request.build_absolute_uri(frame.image.url)
        except Exception:
            image_url = None
        response_data.append(
            {
                "id": frame.id,
                "timestamp": frame.timestamp.isoformat(),
                "distance": dist,
                "image_url": image_url,
            }
        )

    return JsonResponse({"results": response_data})


# ---------------------------------------------------------------------------
# Endpoint: detections for a specific frame (JSON)
# ---------------------------------------------------------------------------


@require_GET
def frame_detections(request, frame_id: int):
    try:
        frame = Frame.objects.get(id=frame_id)
    except Frame.DoesNotExist:
        return JsonResponse({"error": "frame not found"}, status=404)

    detections_qs = frame.detections.all()
    detections = []
    for d in detections_qs:
        detections.append(
            {
                "id": d.id,
                "label": d.label,
                "confidence": d.confidence,
                "bbox": [d.x1, d.y1, d.x2, d.y2],
                "text": d.text,
            }
        )

    return JsonResponse({"detections": detections})
