# Generated by Django 5.2.1 on 2025-06-30 14:02

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("alerts", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="alertrule",
            name="max_distance",
        ),
        migrations.AddField(
            model_name="alertrule",
            name="min_similarity",
            field=models.PositiveSmallIntegerField(
                default=70,
                help_text="Required similarity between rule description and detection embedding (0-100, higher = stricter)",
            ),
        ),
    ]
