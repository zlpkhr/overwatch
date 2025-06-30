# Register your models here.

from django.contrib import admin

from ingest.models import Camera, Frame, DetectedObject

admin.site.register(Camera)
admin.site.register(Frame)
admin.site.register(DetectedObject)
