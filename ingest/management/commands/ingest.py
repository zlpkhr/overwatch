import time
from datetime import datetime

import cv2
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError

from ingest.models import Camera, Frame
from ingest.tasks import detect_objects, generate_embeddings_batch

# Default batch size for embedding/detection jobs
BATCH_SIZE = 8


class Command(BaseCommand):
    help = "Connects to an RTSP stream, extracts frames at 1 FPS, and saves them using Django's storage system."

    def add_arguments(self, parser):
        parser.add_argument(
            "--slug",
            type=str,
            default="default",
            help="Camera slug to ingest (default: 'default')",
        )
        parser.add_argument(
            "--rtsp",
            type=str,
            help="Override RTSP URL (otherwise use camera.rtsp_url)",
        )

    def handle(self, *args, **options):
        # ------------------------------------------------------------------
        # Retrieve camera information
        # ------------------------------------------------------------------

        slug = options.get("slug", "default")
        rtsp_override = options.get("rtsp")

        try:
            camera = Camera.objects.get(slug=slug)
        except Camera.DoesNotExist:
            raise CommandError(f"Camera with slug '{slug}' does not exist.")

        rtsp_url = rtsp_override or camera.rtsp_url or getattr(settings, "RTSP_URL", None)

        if not rtsp_url:
            raise CommandError("RTSP URL missing (camera.rtsp_url or --rtsp or settings.RTSP_URL)")

        self.stdout.write(self.style.NOTICE(f"[{slug}] Connecting to RTSP stream: {rtsp_url}"))

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            self.stdout.write(self.style.ERROR("Failed to open RTSP stream."))
            return

        last_saved = time.time()
        self.stdout.write(
            self.style.NOTICE("Extracting frames at 1 FPS. Press Ctrl+C to stop.")
        )
        try:
            batch = []
            batch_meta = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    self.stdout.write(
                        self.style.WARNING("Empty or invalid frame, exiting...")
                    )
                    break
                now = time.time()
                if now - last_saved >= 1.0:
                    timestamp = datetime.now()
                    filename = f"{timestamp.isoformat(sep='_', timespec='seconds')}.jpg"
                    success, buffer = cv2.imencode(".jpg", frame)
                    if success:
                        batch.append((buffer.tobytes(), filename, timestamp))
                        batch_meta.append((filename, timestamp))
                        self.stdout.write(
                            self.style.SUCCESS(f"Queued frame: {filename}")
                        )
                    else:
                        self.stdout.write(self.style.WARNING("Failed to encode frame."))
                    last_saved = now
                    if len(batch) >= BATCH_SIZE:
                        frame_ids = []
                        for buf, filename, timestamp in batch:
                            frame_instance = Frame.objects.create(
                                image=ContentFile(buf, filename),
                                timestamp=timestamp,
                                camera=camera,
                            )
                            frame_ids.append(frame_instance.id)
                            # schedule detection per frame (can be batched later)
                            detect_objects.delay(frame_instance.id)
                        generate_embeddings_batch.delay(frame_ids)
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"Saved and dispatched batch of {len(batch)} frames."
                            )
                        )
                        batch = []
                        batch_meta = []
            # Process any remaining frames in the last batch
            if batch:
                frame_ids = []
                for buf, filename, timestamp in batch:
                    frame_instance = Frame.objects.create(
                        image=ContentFile(buf, filename),
                        timestamp=timestamp,
                        camera=camera,
                    )
                    frame_ids.append(frame_instance.id)
                    # schedule detection per frame (can be batched later)
                    detect_objects.delay(frame_instance.id)
                generate_embeddings_batch.delay(frame_ids)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Saved and dispatched final batch of {len(batch)} frames."
                    )
                )
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted by user."))
        finally:
            cap.release()
            self.stdout.write(self.style.SUCCESS("Stream closed."))
