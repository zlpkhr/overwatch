# Register your models here.

from django.contrib import admin

from .models import Frame, Camera


@admin.register(Frame)
class FrameAdmin(admin.ModelAdmin):
    list_display = ("id", "timestamp")
    readonly_fields = ("timestamp",)


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "rtsp_url", "created_at")
    prepopulated_fields = {"slug": ("name",)}
    search_fields = ("name", "rtsp_url")
