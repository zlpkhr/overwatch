# Generated by Django 5.2.1 on 2025-06-09 23:46

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("ingest", "0003_camera"),
    ]

    operations = [
        migrations.DeleteModel(
            name="Camera",
        ),
    ]
