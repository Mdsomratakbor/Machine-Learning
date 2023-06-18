from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("./Emotion-Detection-of-Text/emotion_classifier_model_6_april_2023.pkl")

class EmotionRequest(BaseModel):
    text: str


@app.post("/classify_emotion")
def classify_emotion(request: EmotionRequest):
    text = request.text
    # Preprocess the text if needed
    # Perform emotion classification using the loaded model
    predicted_emotion = model.predict(np.array([text]))[0]
       # Create the response JSON
    return {
        'prediction': prediction[0],
        'prediction_score': np.max(prediction_proba),
        'prediction_probabilities': pred_percentage_for_all
    }
