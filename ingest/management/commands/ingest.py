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

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 24
        frame_interval = int(fps)

        frame_count = 0
        self.stdout.write(
            self.style.NOTICE("Extracting frames at 1 FPS. Press Ctrl+C to stop.")
        )
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.stdout.write(
                        self.style.WARNING("Stream ended or cannot fetch frame.")
                    )
                    break
                if frame_count % frame_interval == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"frames/frame_{timestamp}.jpg"
                    success, buffer = cv2.imencode(".jpg", frame)
                    if success:
                        default_storage.save(filename, ContentFile(buffer.tobytes()))
                        self.stdout.write(
                            self.style.SUCCESS(f"Saved frame: {filename}")
                        )
                        time.sleep(1)
                    else:
                        self.stdout.write(self.style.WARNING("Failed to encode frame."))
                frame_count += 1
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted by user."))
        finally:
            cap.release()
            self.stdout.write(self.style.SUCCESS("Stream closed."))
