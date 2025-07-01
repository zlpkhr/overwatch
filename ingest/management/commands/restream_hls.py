import os
import subprocess
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ingest.models import Camera

# ---------------------------------------------------------------------------
# NOTE: This command is a *very* thin wrapper around FFmpeg. It is intended
# for development and small-scale deployments. For production you may want to
# use a dedicated media server (e.g. MediaMTX) or a process supervisor.
# ---------------------------------------------------------------------------

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")


class Command(BaseCommand):
    """Restream **all** cameras to HLS under MEDIA_ROOT/hls/<camera_id>/."""

    help = "Restream every configured RTSP camera to an HLS playlist using FFmpeg"

    def add_arguments(self, parser):
        parser.add_argument(
            "--segment-time",
            type=int,
            default=1,
            help="HLS segment duration in seconds (default: 1)",
        )
        parser.add_argument(
            "--list-size",
            type=int,
            default=5,
            help="Number of segments to keep in playlist (use 0 for unlimited). Default: 5",
        )

    def handle(self, *args, **options):
        cams = list(Camera.objects.all())
        if not cams:
            raise CommandError("No cameras configured. Add some via /cameras/ first.")

        seg_time = int(options.get("segment_time") or options.get("segment-time") or 1)
        list_sz = int(options.get("list_size") or options.get("list-size") or 5)

        self.stdout.write(self.style.NOTICE(f"Starting HLS restream for {len(cams)} cameras…"))

        self._procs = []
        for cam in cams:
            self._spawn_ffmpeg(cam, seg_time, list_sz)

        try:
            self.stdout.write(self.style.SUCCESS("Restreaming – press Ctrl+C to stop"))
            for proc in self._procs:
                proc.wait()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("Interrupted – stopping FFmpeg"))
            for proc in self._procs:
                proc.terminate()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spawn_ffmpeg(self, camera: Camera, segment_time: int, list_size: int):
        from django.conf import settings

        out_dir = Path(settings.MEDIA_ROOT) / "hls" / str(camera.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "index.m3u8"

        cmd = [
            FFMPEG_BIN,
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
            "-i",
            camera.rtsp_url,
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

        self.stdout.write(self.style.NOTICE(f"[Cam {camera.id}] Running: " + " ".join(cmd)))
        proc = subprocess.Popen(cmd)
        self._procs.append(proc)
