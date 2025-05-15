import io

import clip
import torch
from PIL import Image

device = "mps" if torch.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_frame(frame_bytes: bytes) -> list[float]:
    image = preprocess(Image.open(io.BytesIO(frame_bytes))).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).cpu().numpy().tolist()
