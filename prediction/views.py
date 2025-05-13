from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import joblib
import json
import pandas as pd
# from tensorflow.keras.models import load_model
# from keras import metrics

# Load models
rf = joblib.load('prediction/models/best_rf_model.pkl')


@csrf_exempt
def predict_price(request):
    if request.method == 'POST':
        try:
            if request.content_type == 'application/json':
                data = json.loads(request.body)
            else:
                data = json.loads(request.POST.get('json_input', '{}'))

            features = data.get("features", [])
            if len(features) != 109:
                return JsonResponse({'error': 'Expected 109 features.'}, status=400)

            input_array = np.array([features])
            prediction = rf.predict(input_array)

            return JsonResponse({'rf': round(float(prediction[0]), 2)})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return render(request, 'predict_form.html', {'range':range(1,110)})