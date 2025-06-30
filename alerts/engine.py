from __future__ import annotations

import logging
import numpy as np
from asgiref.sync import async_to_sync
from django.core import signals

from ingest.models import DetectedObject

from alerts.models import Alert, AlertRule

logger = logging.getLogger("alerts.engine")


def _cosine_distance(u: list[float], v: list[float]) -> float:
    """Return cosine distance (0 identical —> 2 opposite)."""
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if u_arr.size == 0 or v_arr.size == 0:
        return 1.0  # fallback neutral distance
    denom = (np.linalg.norm(u_arr) * np.linalg.norm(v_arr)) or 1e-8
    similarity = float(u_arr @ v_arr) / denom
    # convert cos similarity to distance (1 - sim)/2 in [0,1]
    return 1 - similarity


def evaluate_object(obj: DetectedObject):
    """Evaluate a single DetectedObject against all active alert rules."""

    logger.info(f"Evaluating object: {obj}")

    for rule in AlertRule.objects.filter(active=True):
        logger.info(
            "Rule %s: label=%s min_conf=%s desc=%s min_sim=%s", rule.id, rule.label, rule.min_confidence, rule.description, rule.min_similarity
        )
        # Label condition -------------------------------------------------------
        if rule.label and obj.label != rule.label:
            logger.info("Skip rule %s due to label mismatch", rule.id)
            continue

        # Confidence threshold --------------------------------------------------
        if obj.confidence < rule.min_confidence:
            logger.info("Skip rule %s due to confidence %.2f < %.2f", rule.id, obj.confidence, rule.min_confidence)
            continue

        # OCR text condition ----------------------------------------------------
        if rule.text_contains:
            if not obj.text or rule.text_contains.lower() not in obj.text.lower():
                logger.info("Skip rule %s due to OCR text mismatch", rule.id)
                continue

        # Embedding similarity --------------------------------------------------
        if rule.embedding:
            if obj.embedding is None:
                logger.info("Skip rule %s – object has no embedding", rule.id)
                continue
            dist = _cosine_distance(obj.embedding, rule.embedding)
            similarity = 1 - dist
            logger.info("Rule %s similarity %.3f", rule.id, similarity)
            if similarity < (rule.min_similarity / 100):
                logger.info("Skip rule %s due to similarity threshold", rule.id)
                continue

        # All conditions satisfied → create alert
        alert = Alert.objects.create(rule=rule, frame=obj.frame, detection=obj)
        logger.info("Created alert %s for rule %s on object %s", alert.id, rule.id, obj.id)
        _notify_clients(alert)


def evaluate_frame(frame):
    """Evaluate an entire frame's embedding against rules (description similarity)."""
    if frame.embedding is None:
        return
    logger.info("Evaluating frame %s", frame.id)
    for rule in AlertRule.objects.filter(active=True):
        if not rule.embedding:
            continue
        similarity = 1 - _cosine_distance(frame.embedding, rule.embedding)
        logger.info("Frame %s rule %s similarity %.3f", frame.id, rule.id, similarity)
        if similarity < (rule.min_similarity / 100):
            continue
        alert = Alert.objects.create(rule=rule, frame=frame, detection=None)
        logger.info("Created frame-level alert %s for rule %s", alert.id, rule.id)
        _notify_clients(alert)


# ---------------------------------------------------------------------------
# Very simple server-side push placeholder (polling, SSE, WS later)
# ---------------------------------------------------------------------------

def _notify_clients(alert: Alert):
    """Emit Django signal so listeners (SSE/WS) or tests can react."""

    async_to_sync(signals.request_started.send)(
        sender=_notify_clients, alert_id=alert.id
    ) 