from django.conf import settings
from django.db import migrations, models
from django.utils import timezone


def forwards(apps, schema_editor):
    Camera = apps.get_model("ingest", "Camera")
    Frame = apps.get_model("ingest", "Frame")

    # Create a default camera so existing frames are kept intact
    default_cam, _created = Camera.objects.get_or_create(
        slug="default",
        defaults={
            "name": "Default Camera",
            "rtsp_url": getattr(settings, "RTSP_URL", ""),
            "is_active": True,
        },
    )

    # Attach all existing frames to the default camera & compute sync_key
    for frame in Frame.objects.filter(camera__isnull=True).iterator():
        frame.camera_id = default_cam.id
        if not frame.sync_key:
            ts = frame.timestamp or timezone.now()
            frame.sync_key = ts.replace(microsecond=0).isoformat()
        frame.save(update_fields=["camera", "sync_key"])


def backwards(apps, schema_editor):
    Camera = apps.get_model("ingest", "Camera")
    # We don't remove frames' camera to avoid dangling references – just delete default cam
    Camera.objects.filter(slug="default").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("ingest", "0007_frame_embedding"),
    ]

    operations = [
        # 1. Create Camera model
        migrations.CreateModel(
            name="Camera",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=100)),
                ("slug", models.SlugField(unique=True)),
                ("rtsp_url", models.CharField(max_length=500)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["name"],
            },
        ),
        # 2. Add new fields to Frame (nullable for data migration)
        migrations.AddField(
            model_name="frame",
            name="camera",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=models.CASCADE,
                related_name="frames",
                to="ingest.camera",
            ),
        ),
        migrations.AddField(
            model_name="frame",
            name="sync_key",
            field=models.CharField(blank=True, db_index=True, max_length=25),
        ),
        # 3. Populate defaults
        migrations.RunPython(forwards, backwards),
        # 4. Optionally set camera to non-nullable in a future migration (kept nullable now for simplicity)
    ]