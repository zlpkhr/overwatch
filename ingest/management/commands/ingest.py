import threading
import time
from datetime import datetime

import cv2
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError

from ingest.models import Camera, Frame
from ingest.tasks import detect_objects, generate_embeddings_batch

BATCH_SIZE = 8


class Command(BaseCommand):
    help = "Ingest all configured RTSP cameras, grab 1 FPS frames, store, and dispatch background tasks."

    def handle(self, *args, **options):
        cameras = list(Camera.objects.all())
        if not cameras:
            raise CommandError("No cameras found. Add cameras via /cameras/ UI or Django shell.")

        self.stdout.write(self.style.NOTICE(f"Starting ingestion for {len(cameras)} cameras…"))

        threads = []
        for cam in cameras:
            t = threading.Thread(target=self._run_camera_loop, args=(cam,), daemon=True)
            t.start()
            threads.append(t)

        # Keep main thread alive
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted by user – stopping ingestion."))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_camera_loop(self, camera):
        self.stdout.write(self.style.NOTICE(f"[Cam {camera.id}] Connecting to {camera.rtsp_url}"))
        cap = cv2.VideoCapture(camera.rtsp_url)
        if not cap.isOpened():
            self.stdout.write(self.style.ERROR(f"[Cam {camera.id}] Failed to open RTSP stream."))
            return

        last_saved = time.time()
        batch = []
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    self.stdout.write(self.style.WARNING(f"[Cam {camera.id}] Empty frame – exiting thread"))
                    break
                now = time.time()
                if now - last_saved >= 1.0:
                    timestamp = datetime.now()
                    filename = f"{camera.id}_{timestamp.isoformat(sep='_', timespec='seconds')}.jpg"
                    success, buffer = cv2.imencode(".jpg", frame)
                    if success:
                        batch.append((buffer.tobytes(), filename, timestamp))
                        self.stdout.write(self.style.SUCCESS(f"[Cam {camera.id}] queued frame {filename}"))
                    last_saved = now
                    if len(batch) >= BATCH_SIZE:
                        self._flush_batch(camera, batch)
                        batch = []
            # flush remaining
            if batch:
                self._flush_batch(camera, batch)
        finally:
            cap.release()
            self.stdout.write(self.style.SUCCESS(f"[Cam {camera.id}] stream closed."))

    def _flush_batch(self, camera, batch):
                frame_ids = []
                for buf, filename, timestamp in batch:
                    frame_instance = Frame.objects.create(
                camera=camera,
                        image=ContentFile(buf, filename),
                        timestamp=timestamp,
                    )
                    frame_ids.append(frame_instance.id)
                    detect_objects.delay(frame_instance.id)
                generate_embeddings_batch.delay(frame_ids)
        self.stdout.write(self.style.SUCCESS(f"[Cam {camera.id}] saved batch of {len(batch)} frames"))
