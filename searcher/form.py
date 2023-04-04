from django import forms


class SearcherForm(forms.Form):
    request = forms.CharField()