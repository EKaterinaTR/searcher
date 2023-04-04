from django.shortcuts import render

from searcher.form import SearcherForm
from searcher.seacher import search


def get_main(request):
    form = SearcherForm()
    return render(request, "searcher/hi.html", {'form': form})


def get_doc(request):
    number = search(request.GET.get('request'))
    return render(request, f"searcher/doc/{number}_doc.txt")

# Create your views here.
