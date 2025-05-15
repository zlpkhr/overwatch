from django.core.management.base import BaseCommand
import cv2
from datetime import datetime
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import time


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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"frames/frame_{timestamp}.jpg"
                    success, buffer = cv2.imencode(".jpg", frame)
                    if success:
                        default_storage.save(filename, ContentFile(buffer.tobytes()))
                        self.stdout.write(
                            self.style.SUCCESS(f"Saved frame: {filename}")
                        )
                    else:
                        self.stdout.write(self.style.WARNING("Failed to encode frame."))
                    last_saved = now
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted by user."))
        finally:
            cap.release()
            self.stdout.write(self.style.SUCCESS("Stream closed."))
