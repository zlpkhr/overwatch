from __future__ import annotations

from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt

from alerts.forms import AlertRuleForm
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
        results.append(
            {
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "rule_name": a.rule.name,
                "label": getattr(a.detection, "label", ""),
                "image_url": image_url,
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
        results.append(
            {
                "id": a.id,
                "timestamp": a.timestamp.isoformat(),
                "rule_name": a.rule.name,
                "label": getattr(a.detection, "label", ""),
                "image_url": image_url,
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
    if request.method == "POST":
        form = AlertRuleForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("alert_rule_list")
    else:
        form = AlertRuleForm()
    return render(request, "alerts/rule_form.html", {"form": form, "is_new": True})


def rule_edit(request, pk: int):
    rule = get_object_or_404(AlertRule, pk=pk)
    if request.method == "POST":
        form = AlertRuleForm(request.POST, instance=rule)
        if form.is_valid():
            form.save()
            return redirect("alert_rule_list")
    else:
        form = AlertRuleForm(instance=rule)
    return render(request, "alerts/rule_form.html", {"form": form, "is_new": False, "rule": rule})


def rule_delete(request, pk: int):
    rule = get_object_or_404(AlertRule, pk=pk)
    if request.method == "POST":
        rule.delete()
        return redirect("alert_rule_list")
    return render(request, "alerts/rule_delete_confirm.html", {"rule": rule}) 