import clip
import numpy as np
import torch

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_query(query: str) -> list[float]:
    text = clip.tokenize(query).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.squeeze(0).cpu().numpy().tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)

    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def compute_similarities(
    query: str, documents: list[dict[str, str | list[float]]]
) -> list[dict[str, str | float]]:
    query_embedding = embed_query(query)

    scores = [
        {
            "key": doc["key"],
            "score": cosine_similarity(query_embedding, doc["embeddings"]),
        }
        for doc in documents
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)

    return scores
