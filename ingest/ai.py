import io

import clip
import torch
from PIL import Image

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_frame(frame_bytes: bytes) -> list[float]:
    image = preprocess(Image.open(io.BytesIO(frame_bytes))).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).cpu().numpy().tolist()


def embed_frames_batch(frames_bytes: list[bytes]) -> list[list[float]]:
    images = [preprocess(Image.open(io.BytesIO(b))) for b in frames_bytes]
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        image_features = model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().tolist()


def embed_text(text: str) -> list[float]:
    """Generate a CLIP embedding for a text string (normalized)."""
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze(0).cpu().numpy().tolist()
