import json
from django.http import HttpResponse
from django.http.response import JsonResponse
from classifier.decorators import require_http_methods
from models import classifiers

# Create your views here.

@require_http_methods(['POST'])
async def classify(request):

    data = json.loads(request.body)

    model = data['model']
    text = data['text']

    if model in classifiers:
        response = {
            'intents': classifiers[model].predict(text)
        }
        print(response)
        return JsonResponse(response)

    return HttpResponse(status=404)