import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # ✅ Fix for deployment platforms like Render

import numpy as np
import librosa
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
import joblib
import tempfile

# Load model & label encoder
MODEL = joblib.load("heart_model_rf.pkl")
ENCODER = joblib.load("label_encoder.pkl")

class PredictHeartSound(APIView):
    parser_classes = [MultiPartParser]

    def get(self, request, format=None):
        return Response({
            "message": "Use POST with a .wav file at this endpoint to get heart condition prediction (normal or abnormal)."
        })

    def post(self, request, format=None):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        if not file.name.endswith(".wav"):
            return Response({"error": "Invalid file format. Please upload a .wav file."}, status=400)

        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(file.read())
                temp_path = temp_audio.name

            # Extract MFCC features
            audio, sr = librosa.load(temp_path, sr=22050)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

            # Predict
            pred = MODEL.predict(mfcc_mean)[0]
            label = ENCODER.inverse_transform([pred])[0]

            os.remove(temp_path)  # Clean up temp file
            return Response({"prediction": label})

        except Exception as e:
            return Response({"error": str(e)}, status=500)
