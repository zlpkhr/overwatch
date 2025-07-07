#!/usr/bin/env python3
"""
Core evaluation script for CLIP-based image-text retrieval.
Extracted from the CCTV surveillance system for benchmarking on Flickr30k dataset.

This script replicates the exact pipeline used in the production CCTV system:
- Same CLIP model and embedding functions from ingest.ai and search.ai
- Same ChromaService setup
- Same YOLO object detection and EasyOCR pipeline
- Same search and deduplication logic
- Measures only recall@10 for performance evaluation

Usage:
    python evaluation_core.py [num_images]

Examples:
    python evaluation_core.py 1      # Test with 1 image
    python evaluation_core.py 500    # Full evaluation with 500 images
    python evaluation_core.py        # Default: 1 image for testing

Requirements:
    - Dataset must be in data/flickr30k-images/ and data/flickr_annotations_30k.csv
    - Uses pandas for CSV parsing and random sampling
    - Includes YOLO object detection and EasyOCR text extraction
    - Evaluates recall@10 only for focused performance assessment
"""

import base64
import io
import json
import logging
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import clip
import numpy as np
import openai
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# YOLO and EasyOCR imports
try:
    import easyocr
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "YOLO and EasyOCR are required. Install with 'pip install ultralytics easyocr'."
    ) from e

# Add parent directory to path to import from the CCTV modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = None
preprocess = None
yolo_model = None
ocr_reader = None

FRAME_ID_TO_PATH: Dict[str, str] = {}  # global mapping for Jina reranker

# SQLite FTS path
FTS_DB_PATH = os.path.join("./evaluation_chroma_data", "text_index.db")

# SQLite FTS connection (persistent)
FTS_CONN: sqlite3.Connection | None = None


def _init_fts():
    global FTS_CONN
    if FTS_CONN is None:
        os.makedirs(os.path.dirname(FTS_DB_PATH), exist_ok=True)
        FTS_CONN = sqlite3.connect(FTS_DB_PATH, timeout=30, check_same_thread=False)
        FTS_CONN.execute("PRAGMA journal_mode=WAL")
        FTS_CONN.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(frame_id, content)"
        )
    return FTS_CONN


# helper for tokenise
import re


def _tokenize(txt: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", txt.lower())


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("evaluation")


def load_models():
    """Load all required models: CLIP, YOLO, and EasyOCR with GPU support"""
    global clip_model, preprocess, yolo_model, ocr_reader

    logger.info("Loading models on device: %s", device)

    # Load CLIP model
    logger.info("Loading CLIP model: ViT-B/32")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    logger.info("CLIP model loaded successfully")

    if YOLO_AVAILABLE:
        # Load YOLO model
        logger.info("Loading YOLO model: yolov8n.pt")
        yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLO model loaded successfully")

        # Load EasyOCR reader with GPU support
        logger.info("Loading EasyOCR reader...")
        use_gpu = device == "cuda"
        ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
        logger.info(f"EasyOCR reader loaded (GPU: {use_gpu})")
    else:
        logger.info("YOLO/EasyOCR not available - using frame-level embeddings only")


def embed_frame(frame_bytes: bytes) -> List[float]:
    """Exact copy of ingest.ai.embed_frame with GPU support"""
    image = preprocess(Image.open(io.BytesIO(frame_bytes))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze(0).cpu().numpy().tolist()


def embed_query(query: str) -> List[float]:
    """Exact copy of search.ai.embed_query with GPU support"""
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze(0).cpu().numpy().tolist()


def detect_objects(frame_bytes: bytes):
    """
    YOLO object detection matching production pipeline
    Returns detections and PIL image
    """
    if not YOLO_AVAILABLE or yolo_model is None:
        return [], None

    # Convert bytes to PIL image
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

    # Run YOLO prediction
    results = yolo_model.predict(np.array(img), imgsz=640, conf=0.25, verbose=False)[0]

    detections = []
    for box, cls_idx, conf in zip(
        results.boxes.xyxy, results.boxes.cls, results.boxes.conf
    ):
        x1, y1, x2, y2 = box.tolist()
        label = yolo_model.names[int(cls_idx)]
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": float(conf),
            }
        )

    return detections, img


def run_ocr(crop_img):
    """
    EasyOCR text extraction matching production pipeline
    """
    if not YOLO_AVAILABLE or ocr_reader is None:
        return None

    result = ocr_reader.readtext(np.array(crop_img))
    # Concatenate strings with high confidence
    texts = [text for _, text, score in result if score > 0.5]
    return " ".join(texts) if texts else None


def normalize_bbox(bbox, width, height):
    """Normalize bounding box coordinates to 0-1 range"""
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def extract_objects_and_embeddings(frame_bytes: bytes, frame_id: str):
    """
    Extract objects and generate embeddings. Returns (embeddings_data, tokens_set)
    tokens_set – unique lowercase alphanum tokens from label+OCR for FTS.
    """
    embeddings_data = []
    tokens_set: set[str] = set()

    # Frame-level embedding (always included)
    frame_embedding = embed_frame(frame_bytes)
    embeddings_data.append(
        {
            "embedding": frame_embedding,
            "id": str(frame_id),
            "metadata": {"frame_id": frame_id, "type": "frame"},
        }
    )

    detections, img = detect_objects(frame_bytes)
    if img and detections:
        width, height = img.size
        for obj_idx, det in enumerate(detections):
            try:
                crop = img.crop(det["bbox"])
                crop_buf = io.BytesIO()
                crop.save(crop_buf, format="JPEG")
                object_embedding = embed_frame(crop_buf.getvalue())
                ocr_text = run_ocr(crop)

                embeddings_data.append(
                    {
                        "embedding": object_embedding,
                        "id": f"{frame_id}_{obj_idx}",
                        "metadata": {
                            "frame_id": frame_id,
                            "type": "object",
                            "label": det["label"],
                            "confidence": det["confidence"],
                            "text": ocr_text or "",
                        },
                    }
                )

                # collect tokens
                for tok in _tokenize(det["label"] + " " + (ocr_text or "")):
                    tokens_set.add(tok)
            except Exception as e:
                logger.error(
                    "Error processing object %s in frame %s: %s", obj_idx, frame_id, e
                )
                continue

    return embeddings_data, tokens_set


# ChromaService implementation matching the production system
class ChromaService:
    _client = None
    _collection = None
    _persist_path = "./evaluation_chroma_data"

    @classmethod
    def get_collection(cls):
        if cls._client is None:
            os.makedirs(cls._persist_path, exist_ok=True)
            cls._client = chromadb.PersistentClient(path=cls._persist_path)
        if cls._collection is None:
            cls._collection = cls._client.get_or_create_collection(
                name="frames", metadata={"hnsw:space": "cosine"}
            )
        return cls._collection

    @classmethod
    def clear_collection(cls):
        """Clear all data from the collection"""
        if cls._client is not None:
            try:
                cls._client.delete_collection("frames")
                cls._collection = None
            except Exception:
                pass  # Collection might not exist


def load_flickr30k_data_pandas(
    images_dir: str, captions_file: str, max_images: int = 1
):
    """
    Load Flickr30k dataset using pandas and random sampling; keeps **one caption per image**.
    """
    logger.info(f"Loading Flickr30k data from {images_dir}")
    logger.info(f"Target sample size: {max_images} images")

    # Load CSV with pandas
    logger.info("Reading captions CSV with pandas...")
    df = pd.read_csv(captions_file)
    logger.info(f"Loaded {len(df)} rows from CSV")
    logger.info(f"Columns: {list(df.columns)}")

    # Parse JSON captions from the 'raw' column
    captions_data = {}
    for idx, row in df.iterrows():
        try:
            # Parse the JSON string in the 'raw' column
            captions_json = json.loads(row["raw"])
            img_filename = row["filename"]
            img_id = img_filename.replace(".jpg", "")

            captions_data[img_id] = captions_json[:1]  # keep only first caption

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing row {idx}: {e}")
            continue

    logger.info(f"Successfully parsed captions for {len(captions_data)} images")

    # Get available images
    image_files = list(Path(images_dir).glob("*.jpg"))
    logger.info(f"Found {len(image_files)} image files in directory")

    # Filter to images that have captions
    valid_image_files = []
    for img_path in image_files:
        img_id = img_path.stem
        if img_id in captions_data:
            valid_image_files.append(img_path)

    logger.info(f"Found {len(valid_image_files)} images with captions")

    # Random sampling
    if len(valid_image_files) > max_images:
        logger.info(
            f"Randomly sampling {max_images} images from {len(valid_image_files)} available"
        )
        valid_image_files = random.sample(valid_image_files, max_images)

    # Build ground truth dictionaries
    image_paths = []
    image_ids = []
    ground_truth = {}
    reverse_ground_truth = {}

    for img_path in valid_image_files:
        img_id = img_path.stem
        captions = captions_data[img_id]

        image_paths.append(str(img_path))
        image_ids.append(img_id)
        FRAME_ID_TO_PATH[img_id] = str(img_path)
        ground_truth[img_id] = captions

        # Build reverse lookup for text-to-image retrieval
        for caption in captions:
            if caption not in reverse_ground_truth:
                reverse_ground_truth[caption] = []
            reverse_ground_truth[caption].append(img_id)

    logger.info(f"Successfully prepared {len(image_ids)} images for evaluation")
    logger.info(f"Total unique captions: {len(reverse_ground_truth)}")

    return image_paths, image_ids, ground_truth, reverse_ground_truth


def index_images_production_pipeline(image_paths: List[str], image_ids: List[str]):
    """
    Index images using the production CCTV pipeline.
    Skips indexing if the same sample has already been indexed (based on meta file).
    """
    meta_path = os.path.join(ChromaService._persist_path, "indexed_meta.json")

    # Check if already indexed
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            prev_ids = meta.get("image_ids", [])
            if set(prev_ids) == set(image_ids):
                FRAME_ID_TO_PATH.update(meta.get("mapping", {}))
                logger.info("Sample already indexed – skipping indexing step.")
                return
        except Exception:
            # malformed meta – reindex
            pass

    logger.info(f"Indexing {len(image_paths)} images using production pipeline…")
    logger.info("Pipeline includes: CLIP embeddings + YOLO detection + EasyOCR")

    # We will reindex – clear existing collection
    ChromaService.clear_collection()

    collection = ChromaService.get_collection()

    total_embeddings = 0
    total_objects = 0

    fts_conn = _init_fts()
    insert_stmt = fts_conn.execute

    for img_path, img_id in tqdm(
        zip(image_paths, image_ids), total=len(image_ids), desc="Indexing"
    ):
        try:
            with open(img_path, "rb") as f:
                frame_bytes = f.read()

            embeddings_data, tokens_set = extract_objects_and_embeddings(
                frame_bytes, img_id
            )

            # Separate for batch insert
            embeddings = [item["embedding"] for item in embeddings_data]
            ids = [item["id"] for item in embeddings_data]
            metadatas = [item["metadata"] for item in embeddings_data]

            object_count = len([m for m in metadatas if m["type"] == "object"])
            total_objects += object_count

            if embeddings:
                collection.upsert(embeddings=embeddings, ids=ids, metadatas=metadatas)
                total_embeddings += len(embeddings)

            FRAME_ID_TO_PATH[img_id] = img_path

            for tok in tokens_set:
                insert_stmt(
                    "INSERT INTO docs (frame_id, content) VALUES (?, ?)",
                    (img_id, tok),
                )

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    fts_conn.commit()
    # Keep connection open for search phase

    logger.info(f"Successfully indexed {total_embeddings} total embeddings")
    logger.info(
        f"Frame embeddings: {len(image_paths)} | Object embeddings: {total_objects}"
    )

    # Write meta for future runs
    try:
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "image_ids": image_ids,
                    "mapping": FRAME_ID_TO_PATH,
                },
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning(f"Warning: Failed to write meta file ({e})")


def search_production_pipeline(
    query: str, n_results: int = 10
) -> List[Tuple[str, float]]:
    """Full production search: LLM expansion, Chroma, Jina reranker."""
    # 1) Expand query via GPT (fallback to original)
    expansions = expand_query_llm(query, n_expansions=3)

    collection = ChromaService.get_collection()
    all_results: List[Tuple[str, float]] = []

    # 2) For each expansion, embed and search Chroma
    for q in expansions:
        emb = embed_query(q)
        res = collection.query(query_embeddings=[emb], n_results=n_results)
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = (
            res.get("metadatas", [[]])[0] if res.get("metadatas") else [{}] * len(ids)
        )
        for _id, dist, meta in zip(ids, dists, metas):
            frame_id = str(meta.get("frame_id") or _id.split("_", 1)[0])
            all_results.append((frame_id, dist))

    # 3) Deduplicate keeping best distance
    best: Dict[str, float] = {}
    for fid, dist in all_results:
        if fid not in best or dist < best[fid]:
            best[fid] = dist

    # 4) Sort by distance and take top candidates
    sorted_dists = sorted(best.items(), key=lambda x: x[1])
    top_candidates = sorted_dists[:n_results]

    # 5) Prepare images for reranker
    images_for_rerank = []
    id_order = []
    for fid, _ in top_candidates:
        path = FRAME_ID_TO_PATH.get(fid)
        if path:
            b64 = encode_image_to_base64(path)
            images_for_rerank.append({"image": b64})
            id_order.append(fid)

    # 6) Call Jina reranker (fall back if fails)
    reranked_idx, _scores = call_jina_reranker(query, images_for_rerank, n_results)

    # 7) Build final ordered list following reranked indices
    final_list: List[Tuple[str, float]] = []
    for idx in reranked_idx:
        if idx < len(id_order):
            fid = id_order[idx]
            final_list.append((fid, best[fid]))
    # Safety: append any missing ids in original order
    remaining = [fid for fid in id_order if fid not in {id for id, _ in final_list}]
    for fid in remaining:
        final_list.append((fid, best[fid]))

    # 4b) Add frames matched by keyword tokens
    for q in expansions:
        for tok in _tokenize(q):
            for fid in (
                _init_fts()
                .execute("SELECT frame_id FROM docs WHERE content = ?", (tok,))
                .fetchall()
            ):
                if fid[0] not in best:
                    best[fid[0]] = 0.9  # heuristic distance for text match

    return final_list


def evaluate_metrics(reverse_ground_truth: Dict[str, List[str]], max_workers: int = 5) -> Tuple[float, float]:
    """Compute Recall@10 and MRR using limited parallelism."""
    captions = list(reverse_ground_truth.keys())
    total_queries = len(captions)
    recall_sum = 0.0
    mrr_sum = 0.0

    def _process(caption: str):
        relevant = set(reverse_ground_truth[caption])
        res = search_production_pipeline(caption, n_results=100)
        retrieved_ids = [fid for fid, _ in res]
        # Recall@10
        hit = len(relevant.intersection(set(retrieved_ids[:10])))
        # MRR
        rr = 0.0
        for rank, fid in enumerate(retrieved_ids, 1):
            if fid in relevant:
                rr = 1.0 / rank
                break
        return hit, len(relevant), rr

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, cap): cap for cap in captions}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=total_queries, desc="Eval R@10 & MRR"):
            hit, denom, rr = fut.result()
            if denom:
                recall_sum += hit / denom
            mrr_sum += rr

    recall_10 = recall_sum / total_queries if total_queries else 0.0
    mrr = mrr_sum / total_queries if total_queries else 0.0
    return recall_10, mrr


def run_production_evaluation():
    """
    Run complete evaluation using the exact production CCTV pipeline.
    """
    logger.info("=== CCTV Production Pipeline Evaluation ===")
    logger.info(f"Using device: {device}")
    logger.info("")

    # Load all models
    load_models()

    # Configuration - matches production setup
    images_dir = "data/flickr30k-images"
    captions_file = "data/flickr_annotations_30k.csv"

    # Fixed sample size
    max_images = 10

    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Captions file: {captions_file}")
    logger.info(f"Sample size: {max_images} images")
    logger.info("")

    # Load data using pandas
    image_paths, image_ids, ground_truth, reverse_ground_truth = (
        load_flickr30k_data_pandas(images_dir, captions_file, max_images)
    )

    if not reverse_ground_truth:
        logger.error("No captions loaded! Check your data files.")
        return

    # Index images using production pipeline
    index_images_production_pipeline(image_paths, image_ids)

    # Run evaluation
    logger.info("\n=== Running Production Pipeline Evaluation ===")

    # Evaluate both metrics in one pass
    recall_10, mrr = evaluate_metrics(reverse_ground_truth)

    # Print results
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"Dataset sample size: {max_images} images")
    logger.info(f"Total captions evaluated: {len(reverse_ground_truth)}")
    logger.info(f"Images with captions: {len(ground_truth)}")
    logger.info("")
    logger.info(f"Recall@10: {recall_10:.4f}")
    logger.info(f"MRR: {mrr:.4f}")

    # Save results
    results = {
        "recall_at_10": recall_10,
        "mrr": mrr,
        "dataset_info": {
            "sample_size": max_images,
            "total_captions": len(reverse_ground_truth),
            "images_with_captions": len(ground_truth),
        },
        "pipeline_info": {
            "model": "ViT-B/32",
            "device": device,
            "yolo_available": YOLO_AVAILABLE,
            "embedding_functions": "production CCTV pipeline",
            "search_method": "ChromaService with cosine distance",
            "includes_object_detection": YOLO_AVAILABLE,
            "includes_ocr": YOLO_AVAILABLE,
        },
    }

    output_file = f"evaluation_results_production_{max_images}images.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


# ---------------------------------------------------------------------------
# LLM query expansion and Jina reranker helpers (from search/views.py)
# ---------------------------------------------------------------------------

EXPANSION_PROMPT = """
Expand a user query for a CCTV footage analysis and search tool to generate multiple variations tailored to finding relevant frames. The number of expanded queries should match the user's request.

Steps:
1. Understand the user query and what they are searching for in the CCTV footage.
2. Generate multiple variations of the query, focusing on different aspects that might appear in the footage (such as angles, times, related objects, or actions).
3. Ensure the expanded queries maintain the original search intent and help retrieve all relevant frames.
4. The number of expanded queries should match the user-specified amount (n).

Output format:
Return the expanded queries as json object with a single key \"queries\" and a list of natural language strings as value.
"""


def expand_query_llm(query: str, n_expansions: int = 3) -> List[str]:
    """Replicates search.views.expand_query_llm; falls back gracefully."""
    try:
        # Uses environment variable OPENAI_API_KEY
        client = openai.OpenAI()
        response = client.responses.create(
            model="gpt-4.1",
            instructions=EXPANSION_PROMPT,
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
            store=False,
        )
        data = json.loads(response.output_text)
        return data.get("queries", [query]) or [query]
    except Exception as e:
        logger.error(f"expand_query_llm error: {e} (fallback to original query)")
        return [query]


def encode_image_to_base64(path: str) -> str:
    """Read a file and encode it as base64 string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"encode_image_to_base64 error for {path}: {e}")
        return ""


def call_jina_reranker(
    query: str, images_for_rerank: List[Dict[str, str]], n_results: int
):
    """Replicates search.views.call_jina_reranker."""
    reranked_indices = list(range(len(images_for_rerank)))
    rerank_scores = None
    if not images_for_rerank:
        return reranked_indices, rerank_scores
    try:
        jina_key = os.getenv("JINA_API_KEY")
        if not jina_key:
            raise RuntimeError("JINA_API_KEY env var not set")
        headers = {
            "Authorization": f"Bearer {jina_key}",
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
            "https://api.jina.ai/v1/rerank", headers=headers, json=payload, timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            sorted_res = sorted(data["results"], key=lambda x: -x["relevance_score"])
            reranked_indices = [r["index"] for r in sorted_res]
            rerank_scores = [r["relevance_score"] for r in sorted_res]
    except Exception as e:
        logger.error(f"call_jina_reranker error: {e} (fallback to original order)")
    return reranked_indices, rerank_scores


if __name__ == "__main__":
    # Set random seed for reproducible sampling
    random.seed(42)
    np.random.seed(42)

    run_production_evaluation()
