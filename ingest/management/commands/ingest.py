import time
from datetime import datetime

import cv2
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand

from ingest.models import Frame
from ingest.tasks import generate_embeddings


class Command(BaseCommand):
    help = "Connects to an RTSP stream, extracts frames at 1 FPS, and saves them using Django's storage system."

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.NOTICE(f"Connecting to RTSP stream: {settings.RTSP_URL}")
        )

        cap = cv2.VideoCapture(settings.RTSP_URL)
        if not cap.isOpened():
            self.stdout.write(self.style.ERROR("Failed to open RTSP stream."))
            return

        last_saved = time.time()
        self.stdout.write(
            self.style.NOTICE("Extracting frames at 1 FPS. Press Ctrl+C to stop.")
        )
        try:
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
                        frame_instance = Frame.objects.create(
                            image=ContentFile(buffer.tobytes(), filename),
                            timestamp=timestamp,
                        )
                        generate_embeddings.delay(frame_instance.id)
                        self.stdout.write(
                            self.style.SUCCESS(f"Saved frame: {frame_instance}")
                        )
                    else:
                        self.stdout.write(self.style.WARNING("Failed to encode frame."))
                    last_saved = now
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted by user."))
        finally:
            cap.release()
            self.stdout.write(self.style.SUCCESS("Stream closed."))
