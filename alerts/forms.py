from django import forms

from alerts.models import AlertReferenceImage, AlertRule


class AlertRuleForm(forms.ModelForm):
    class Meta:
        model = AlertRule
        fields = [
            "name",
            "active",
            "description",
            "min_similarity",
        ]
        widgets = {
            "min_similarity": forms.NumberInput(
                attrs={"type": "range", "min": 0, "max": 100, "step": 1}
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.instance.pk and not self.fields["min_similarity"].initial:
            self.fields["min_similarity"].initial = 15


class AlertReferenceImageForm(forms.ModelForm):
    class Meta:
        model = AlertReferenceImage
        fields = ["image"]
        widgets = {
            "image": forms.ClearableFileInput(
                attrs={"class": "form-control", "accept": "image/*", "multiple": False}
            ),
        }
