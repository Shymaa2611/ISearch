from django import forms
from .models import SearchEngineModel

class SearchForm(forms.ModelForm):
    class Meta:
        model=SearchEngineModel
        fields=('query','model')
    


