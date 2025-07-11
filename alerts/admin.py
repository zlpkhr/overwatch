from django.contrib import admin

from alerts.models import Alert, AlertReferenceImage, AlertRule


@admin.register(AlertRule)
class AlertRuleAdmin(admin.ModelAdmin):
    list_display = ("name", "active", "description", "min_similarity")
    list_filter = ("active", "label")
    search_fields = ("name", "label", "text_contains")


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "rule", "frame", "acknowledged")
    list_filter = ("acknowledged", "rule__name")
    date_hierarchy = "timestamp"


@admin.register(AlertReferenceImage)
class AlertReferenceImageAdmin(admin.ModelAdmin):
    list_display = ("id", "rule", "created_at")
    search_fields = ("rule__name",)
    readonly_fields = ("embedding",)
