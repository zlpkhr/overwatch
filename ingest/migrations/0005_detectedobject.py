# Generated by Django 5.2.1 on 2025-06-10 00:36

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ingest", "0004_delete_camera"),
    ]

    operations = [
        migrations.CreateModel(
            name="DetectedObject",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("label", models.CharField(max_length=100)),
                ("confidence", models.FloatField()),
                ("x1", models.FloatField()),
                ("y1", models.FloatField()),
                ("x2", models.FloatField()),
                ("y2", models.FloatField()),
                ("text", models.TextField(blank=True, null=True)),
                ("embedding", models.JSONField(blank=True, null=True)),
                (
                    "frame",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="objects",
                        to="ingest.frame",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(fields=["label"], name="ingest_dete_label_e8cbd5_idx")
                ],
            },
        ),
    ]
