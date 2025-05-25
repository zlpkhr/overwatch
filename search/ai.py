import clip
import torch

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_query(query: str) -> list[float]:
    text = clip.tokenize(query).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.squeeze(0).cpu().numpy().tolist()
