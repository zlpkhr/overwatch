# Create your views here.

import json

import openai
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ingest.models import Frame
from search.ai import embed_query
from search.chroma_service import ChromaService

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


@require_GET
def search_frames(request):
    query = request.GET.get("q")
    n_results = int(request.GET.get("n", 10))

    if not query:
        return JsonResponse({"error": "Missing query parameter q"}, status=400)

    # LLM-based multi-query expansion
    try:
        expanded_queries = expand_query_llm(query, n_expansions=3)
    except Exception:
        # fallback to original query if LLM fails
        expanded_queries = [query]

    # Embed all expanded queries
    query_embeddings = [embed_query(q) for q in expanded_queries]
    collection = ChromaService.get_collection()
    all_results = []
    for emb in query_embeddings:
        results = collection.query(query_embeddings=[emb], n_results=n_results)
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        all_results.extend(zip(ids, distances))
    # Deduplicate by id, keep best (lowest) distance
    best = {}
    for frame_id, dist in all_results:
        if frame_id not in best or dist < best[frame_id]:
            best[frame_id] = dist
    # Sort by distance
    sorted_results = sorted(best.items(), key=lambda x: x[1])
    if not sorted_results:
        return JsonResponse({"results": []})
    min_dist = sorted_results[0][1]
    max_dist = sorted_results[-1][1]
    range_dist = max_dist - min_dist if max_dist != min_dist else 1.0
    response = []
    for frame_id, dist in sorted_results[:n_results]:
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
