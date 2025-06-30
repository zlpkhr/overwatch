import os
import subprocess
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from ingest.models import Camera

# ---------------------------------------------------------------------------
# NOTE: This command is a *very* thin wrapper around FFmpeg. It is intended
# for development and small-scale deployments. For production you may want to
# use a dedicated media server (e.g. MediaMTX) or a process supervisor.
# ---------------------------------------------------------------------------

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")


class Command(BaseCommand):
    """Restream all active Cameras as HLS under MEDIA_ROOT/hls/<slug>/."""

    help = "Restream every active camera. No arguments."

    def handle(self, *args, **options):
        cameras = Camera.objects.filter(is_active=True)

        # Auto-create default camera if DB empty and settings.RTSP_URL present
        if not cameras.exists():
            default_url = getattr(settings, "RTSP_URL", None)
            if default_url:
                cam = Camera.objects.create(
                    name="Default Camera",
                    slug="overwatch",
                    rtsp_url=default_url,
                    is_active=True,
                )
                cameras = Camera.objects.filter(pk=cam.pk)
            else:
                raise CommandError("No cameras configured and settings.RTSP_URL missing.")

        # HLS params (hard-coded defaults)
        segment_time = 1
        list_size = 5

        self._procs = []

        for cam in cameras:
            self.stdout.write(
                self.style.NOTICE(f"Restreaming camera '{cam.slug}' -> {cam.rtsp_url}")
            )
            self._spawn_ffmpeg(cam.slug, cam.rtsp_url, segment_time, list_size)

        # Wait for all processes
        try:
            self.stdout.write(
                self.style.SUCCESS("Restreaming – press Ctrl+C to stop")
            )
            for proc in self._procs:
                proc.wait()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted – stopping FFmpeg"))
            for proc in self._procs:
                proc.terminate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _spawn_ffmpeg(
        self, slug: str, rtsp_url: str, segment_time: int, list_size: int
    ):
        out_dir = Path(settings.MEDIA_ROOT) / "hls" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "index.m3u8"

        # Build FFmpeg command
        # We copy the video stream to avoid re-encoding when possible.
        # Audio is encoded to AAC to maximise compatibility.
        cmd = [
            FFMPEG_BIN,
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
            "-i",
            rtsp_url,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-f",
            "hls",
            "-hls_time",
            str(segment_time),
            "-hls_list_size",
            str(list_size),
            "-hls_flags",
            "append_list+independent_segments+program_date_time",
            str(out_path),
        ]

        self.stdout.write(self.style.NOTICE("Running: " + " ".join(cmd)))

        # Spawn as a subprocess. We store references so we can terminate later.
        proc = subprocess.Popen(cmd)
        if not hasattr(self, "_procs"):
            self._procs = []
        self._procs.append(proc)
