from django.urls import path

from alerts import views


urlpatterns = [
    # API endpoints
    path("api/alerts/unacked/", views.unacked_alerts, name="api_unacked_alerts"),
    path("api/alerts/<int:pk>/ack/", views.ack_alert, name="api_ack_alert"),
    path("api/alerts/recent/", views.recent_alerts, name="api_recent_alerts"),

    # Alerts dashboard
    path("alerts/", views.alerts_page, name="alerts_page"),

    # Rule management
    path("alerts/rules/", views.rule_list, name="alert_rule_list"),
    path("alerts/rules/new/", views.rule_new, name="alert_rule_new"),
    path("alerts/rules/<int:pk>/edit/", views.rule_edit, name="alert_rule_edit"),
    path("alerts/rules/<int:pk>/delete/", views.rule_delete, name="alert_rule_delete"),
] 