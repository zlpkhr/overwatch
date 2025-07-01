from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from alerts.forms import AlertReferenceImageForm, AlertRuleForm
from alerts.models import Alert, AlertRule

# ---------------------------------------------------------------------------
# Alert feed + page
# ---------------------------------------------------------------------------


@require_GET
def unacked_alerts(request):
    """Return latest unacknowledged alerts (JSON)."""

    qs = Alert.objects.filter(acknowledged=False).select_related("rule", "frame")[:20]
    results = []
    for a in qs:
        try:
            image_url = request.build_absolute_uri(a.frame.image.url)
        except Exception:
            image_url = None
        player_url = (
            request.build_absolute_uri(
                reverse("frame_player") + f"?ts={a.frame.timestamp.isoformat()}"
            )
            if a.frame
            else None
        )
        results.append(
            {
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "rule_name": a.rule.name,
                "label": getattr(a.detection, "label", ""),
                "image_url": image_url,
                "frame_id": a.frame.id if a.frame else None,
                "camera_id": a.frame.camera_id if a.frame else None,
                "frame_player_url": player_url,
            }
        )
    return JsonResponse({"results": results})


@csrf_exempt
@require_POST
def ack_alert(request, pk: int):
    """Mark alert as acknowledged."""

    alert = get_object_or_404(Alert, pk=pk)
    alert.acknowledged = True
    alert.save()
    return JsonResponse({"ok": True})


def alerts_page(request):
    """Render the alerts dashboard."""

    return render(request, "alerts/alerts.html")


@require_GET
def recent_alerts(request):
    """Return recent alerts irrespective of acknowledged state.

    Query parameter:
        limit  – how many alerts to return (default 50)
    """

    limit = int(request.GET.get("limit", 50))
    qs = Alert.objects.select_related("rule", "frame").order_by("-timestamp")[:limit]
    results = []
    for a in qs:
        try:
            image_url = request.build_absolute_uri(a.frame.image.url)
        except Exception:
            image_url = None
        player_url = (
            request.build_absolute_uri(
                reverse("frame_player") + f"?ts={a.frame.timestamp.isoformat()}"
            )
            if a.frame
            else None
        )
        results.append(
            {
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "rule_name": a.rule.name,
                "label": getattr(a.detection, "label", ""),
                "image_url": image_url,
                "frame_id": a.frame.id if a.frame else None,
                "camera_id": a.frame.camera_id if a.frame else None,
                "frame_player_url": player_url,
                "acknowledged": a.acknowledged,
            }
        )
    return JsonResponse({"results": results})


# ---------------------------------------------------------------------------
# AlertRule CRUD
# ---------------------------------------------------------------------------


def rule_list(request):
    return render(request, "alerts/rule_list.html", {"rules": AlertRule.objects.all()})


def rule_new(request):
    """Create a new alert rule and optionally upload reference images in same form."""

    if request.method == "POST":
        form = AlertRuleForm(request.POST)
        if form.is_valid():
            rule = form.save()

            # Handle uploaded reference images (may be multiple)
            from alerts.models import AlertReferenceImage

            for file in request.FILES.getlist("reference_images"):
                AlertReferenceImage.objects.create(rule=rule, image=file)

            return redirect("alert_rule_list")
    else:
        form = AlertRuleForm()

    return render(
        request,
        "alerts/rule_form.html",
        {"form": form, "is_new": True},
    )


def rule_edit(request, pk: int):
    rule = get_object_or_404(AlertRule, pk=pk)
    if request.method == "POST":
        form = AlertRuleForm(request.POST, instance=rule)
        if form.is_valid():
            form.save()
            return redirect("alert_rule_list")
    else:
        form = AlertRuleForm(instance=rule)
    return render(
        request, "alerts/rule_form.html", {"form": form, "is_new": False, "rule": rule}
    )


def rule_delete(request, pk: int):
    rule = get_object_or_404(AlertRule, pk=pk)
    if request.method == "POST":
        rule.delete()
        return redirect("alert_rule_list")
    return render(request, "alerts/rule_delete_confirm.html", {"rule": rule})


# ---------------------------------------------------------------------------
# Reference images management for a rule
# ---------------------------------------------------------------------------


def rule_images(request, pk: int):
    """List and upload reference images for an AlertRule."""

    rule = get_object_or_404(AlertRule, pk=pk)

    if request.method == "POST":
        form = AlertReferenceImageForm(request.POST, request.FILES)
        if form.is_valid():
            ref_img = form.save(commit=False)
            ref_img.rule = rule
            ref_img.save()
            return HttpResponseRedirect(request.path_info)
    else:
        form = AlertReferenceImageForm()

    return render(
        request,
        "alerts/rule_images.html",
        {"rule": rule, "form": form, "images": rule.reference_images.all()},
    )
