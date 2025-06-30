# ---------------------------------------------------------------------------
# Object detection + OCR helpers (YOLOv8 + EasyOCR)
# ---------------------------------------------------------------------------
import io

import numpy as np
from celery import shared_task
from PIL import Image

from ingest.ai import embed_frame, embed_frames_batch
from ingest.models import DetectedObject, Frame
from search.chroma_service import ChromaService

try:
    import easyocr
    from ultralytics import YOLO

    _yolo_model = YOLO("yolov8n.pt")  # tiny model
    _ocr_reader = easyocr.Reader(["en"], gpu=False)
except Exception:
    _yolo_model = None
    _ocr_reader = None


def _load_models():
    global _yolo_model, _ocr_reader
    if _yolo_model is None:
        from ultralytics import YOLO

        _yolo_model = YOLO("yolov8n.pt")
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"], gpu=False)


def _detect_objects(frame_bytes):
    _load_models()
    # YOLO expects file path or numpy image
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    res = _yolo_model.predict(np.array(img), imgsz=640, conf=0.25, verbose=False)[0]
    detections = []
    for box, cls_idx, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        label = _yolo_model.names[int(cls_idx)]
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": float(conf),
            }
        )
    return detections, img


def _run_ocr(crop_img):
    _load_models()
    result = _ocr_reader.readtext(np.array(crop_img))
    # concatenate strings with high confidence
    texts = [text for _, text, score in result if score > 0.5]
    return " ".join(texts) if texts else None


def _normalize_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def extract_objects_and_save(frame: Frame):
    bytes_data = frame.image.read()
    detections, img = _detect_objects(bytes_data)
    width, height = img.size
    collection = ChromaService.get_collection()

    for det in detections:
        nx1, ny1, nx2, ny2 = _normalize_bbox(det["bbox"], width, height)
        # crop and embed
        crop = img.crop(det["bbox"])
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        embedding = embed_frame(buf.getvalue())

        # OCR on crop
        text = _run_ocr(crop)

        obj = DetectedObject.objects.create(
            frame=frame,
            label=det["label"],
            confidence=det["confidence"],
            x1=nx1,
            y1=ny1,
            x2=nx2,
            y2=ny2,
            text=text,
            embedding=embedding,
        )

        # upsert into Chroma using composite id frameid_objid
        collection.upsert(
            embeddings=[embedding],
            ids=[f"{frame.id}_{obj.id}"],
            metadatas=[{"frame_id": frame.id, "label": obj.label}],
        )


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------


@shared_task
def generate_embeddings(frame_id: int):
    frame = Frame.objects.get(id=frame_id)
    embeddings = embed_frame(frame.image.read())
    # save embedding to frame
    frame.embedding = embeddings
    frame.save(update_fields=["embedding"])
    collection = ChromaService.get_collection()
    collection.upsert(
        embeddings=[embeddings],
        ids=[str(frame.id)],
        metadatas=[{"timestamp": str(frame.timestamp)}],
    )
    # evaluate alert rules based on frame embedding
    try:
        from alerts.engine import evaluate_frame

        evaluate_frame(frame)
    except Exception:
        pass


@shared_task
def generate_embeddings_batch(frame_ids: list[int]):
    frames = list(Frame.objects.filter(id__in=frame_ids))
    images_bytes = [frame.image.read() for frame in frames]
    embeddings_list = embed_frames_batch(images_bytes)
    collection = ChromaService.get_collection()
    for frame, embedding in zip(frames, embeddings_list):
        collection.upsert(
            embeddings=[embedding],
            ids=[str(frame.id)],
            metadatas=[{"timestamp": str(frame.timestamp)}],
        )
        frame.embedding = embedding
        frame.save(update_fields=["embedding"])
        try:
            from alerts.engine import evaluate_frame
            evaluate_frame(frame)
        except Exception:
            pass


# Object detection/OCR task ---------------------------------------------------


@shared_task
def detect_objects(frame_id: int):
    frame = Frame.objects.get(id=frame_id)
    extract_objects_and_save(frame)
