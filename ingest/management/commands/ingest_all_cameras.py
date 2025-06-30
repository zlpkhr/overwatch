import subprocess
import sys

from django.core.management.base import BaseCommand

from ingest.models import Camera


class Command(BaseCommand):
    """Spawn one ingest process per active camera.

    Each subprocess simply runs `python manage.py ingest --slug=<camera.slug>`.
    This is a minimal orchestration layer – for production you should use a
    real process supervisor (systemd, supervisord, Docker, etc.).
    """

    help = "Start ingestion for all active cameras (development helper)."

    def handle(self, *args, **options):
        cameras = list(Camera.objects.filter(is_active=True))
        if not cameras:
            self.stdout.write(self.style.ERROR("No active cameras found."))
            return

        procs: list[subprocess.Popen] = []
        try:
            for cam in cameras:
                cmd = [sys.executable, "manage.py", "ingest", f"--slug={cam.slug}"]
                self.stdout.write(self.style.NOTICE(f"Launching ingest for camera '{cam.slug}'"))
                procs.append(subprocess.Popen(cmd))

            # Wait for all subprocesses (Ctrl+C to stop)
            self.stdout.write(self.style.SUCCESS("Ingestion processes started – press Ctrl+C to stop"))
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted – terminating workers"))
            for p in procs:
                p.terminate()