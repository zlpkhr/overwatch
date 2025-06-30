from __future__ import annotations

import logging

import numpy as np
from django.core import signals

from alerts.models import Alert, AlertRule
from ingest.models import DetectedObject

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
            "Rule %s: label=%s min_conf=%s desc=%s min_sim=%s",
            rule.id,
            rule.label,
            rule.min_confidence,
            rule.description,
            rule.min_similarity,
        )
        # Label condition -------------------------------------------------------
        if rule.label and obj.label != rule.label:
            logger.info("Skip rule %s due to label mismatch", rule.id)
            continue

        # Confidence threshold --------------------------------------------------
        if obj.confidence < rule.min_confidence:
            logger.info(
                "Skip rule %s due to confidence %.2f < %.2f",
                rule.id,
                obj.confidence,
                rule.min_confidence,
            )
            continue

        # OCR text condition ----------------------------------------------------
        if rule.text_contains:
            if not obj.text or rule.text_contains.lower() not in obj.text.lower():
                logger.info("Skip rule %s due to OCR text mismatch", rule.id)
                continue

        # Embedding similarity --------------------------------------------------
        # Priority: reference images > description embedding fallback
        if rule.reference_images.exists() or rule.embedding:
            if obj.embedding is None:
                logger.info("Skip rule %s – object has no embedding", rule.id)
                continue

            similarities = []
            # Compare with each reference image embedding (if any)
            for ref in rule.reference_images.all():
                if ref.embedding:
                    dist = _cosine_distance(obj.embedding, ref.embedding)
                    similarities.append(1 - dist)

            # If no reference images or they lacked embedding, fall back to rule.embedding
            if not similarities and rule.embedding:
                dist = _cosine_distance(obj.embedding, rule.embedding)
                similarities.append(1 - dist)

            best_sim = max(similarities) if similarities else 0.0
            logger.info("Rule %s best similarity %.3f", rule.id, best_sim)
            if best_sim < (rule.min_similarity / 100):
                logger.info("Skip rule %s due to similarity threshold", rule.id)
                continue

        # All conditions satisfied → create alert
        alert = Alert.objects.create(rule=rule, frame=obj.frame, detection=obj)
        logger.info(
            "Created alert %s for rule %s on object %s", alert.id, rule.id, obj.id
        )
        _notify_clients(alert)


def evaluate_frame(frame):
    """Evaluate an entire frame's embedding against rules (description similarity)."""
    if frame.embedding is None:
        return
    logger.info("Evaluating frame %s", frame.id)
    for rule in AlertRule.objects.filter(active=True):
        if not (rule.reference_images.exists() or rule.embedding):
            continue

        similarities = []
        # Reference images first
        for ref in rule.reference_images.all():
            if ref.embedding:
                similarities.append(
                    1 - _cosine_distance(frame.embedding, ref.embedding)
                )

        if not similarities and rule.embedding:
            similarities.append(1 - _cosine_distance(frame.embedding, rule.embedding))

        best_sim = max(similarities) if similarities else 0.0
        logger.info(
            "Frame %s rule %s best similarity %.3f", frame.id, rule.id, best_sim
        )
        if best_sim < (rule.min_similarity / 100):
            continue
        alert = Alert.objects.create(rule=rule, frame=frame, detection=None)
        logger.info("Created frame-level alert %s for rule %s", alert.id, rule.id)
        _notify_clients(alert)


# ---------------------------------------------------------------------------
# Very simple server-side push placeholder (polling, SSE, WS later)
# ---------------------------------------------------------------------------


def _notify_clients(alert: Alert):
    """Emit Django signal so listeners (SSE/WS) or tests can react."""
    # Send signal synchronously; wrapping with async_to_sync caused errors when
    # called inside Celery tasks (returned list incorrectly awaited).
    signals.request_started.send(sender=_notify_clients, alert_id=alert.id)
