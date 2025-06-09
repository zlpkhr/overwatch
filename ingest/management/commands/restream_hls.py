import os
import subprocess
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

# ---------------------------------------------------------------------------
# NOTE: This command is a *very* thin wrapper around FFmpeg. It is intended
# for development and small-scale deployments. For production you may want to
# use a dedicated media server (e.g. MediaMTX) or a process supervisor.
# ---------------------------------------------------------------------------

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")


class Command(BaseCommand):
    """Restream one (or all) Cameras as HLS under MEDIA_ROOT/hls/<slug>/.

    Usage:
        python manage.py restream_hls --slug=<camera-slug>
        python manage.py restream_hls               # restream *all* cameras

    The command will spawn a long-running FFmpeg process per camera.
    We *do not* daemonise – use a process manager (systemd / supervisord) or
    Docker to keep this running in production.
    """

    help = "Restream RTSP cameras to HLS using FFmpeg"

    def add_arguments(self, parser):
        parser.add_argument(
            "--slug",
            type=str,
            default="live",
            help="Slug to use for the HLS output directory (default: 'live')",
        )
        parser.add_argument(
            "--rtsp",
            type=str,
            help="Optional RTSP URL to restream (overrides settings.RTSP_URL)",
        )

    def handle(self, *args, **options):
        slug = options.get("slug", "live")
        rtsp_url = options.get("rtsp") or settings.RTSP_URL

        if not rtsp_url:
            raise CommandError("RTSP URL must be provided via --rtsp or settings.RTSP_URL")

        self.stdout.write(self.style.NOTICE(f"Starting HLS restream (slug '{slug}')"))
        self._spawn_ffmpeg(slug, rtsp_url)

        # Keep process alive (Ctrl+C to exit)
        try:
            self.stdout.write(self.style.SUCCESS("Restreaming – press Ctrl+C to stop"))
            for proc in self._procs:
                proc.wait()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted – stopping FFmpeg"))
            for proc in self._procs:
                proc.terminate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _spawn_ffmpeg(self, slug: str, rtsp_url: str):
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
            "1",
            "-hls_list_size",
            "5",
            "-hls_flags",
            "delete_segments+append_list+independent_segments",
            str(out_path),
        ]

        self.stdout.write(self.style.NOTICE("Running: " + " ".join(cmd)))

        # Spawn as a subprocess. We store references so we can terminate later.
        proc = subprocess.Popen(cmd)
        if not hasattr(self, "_procs"):
            self._procs = []
        self._procs.append(proc) 