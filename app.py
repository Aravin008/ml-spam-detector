from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Load model and vectorizer
model_path = "models/spam_model.pkl"
vectorizer_path = "models/vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise RuntimeError("Model or vectorizer not found. Run train_model.py first.")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Init FastAPI
app = FastAPI(title="Spam Detector API")

# Input format
class MessageRequest(BaseModel):
    message: str

# Endpoint
@app.post("/predict")
def predict_spam(request: MessageRequest):
    try:
        msg_vector = vectorizer.transform([request.message])
        prediction = model.predict(msg_vector)[0]
        return {
            "message": request.message,
            "prediction": "spam" if prediction == "spam" else "ham"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Feedback API
from typing import Literal
import csv
import datetime

class FeedbackRequest(BaseModel):
    message: str
    correct_label: Literal["spam", "ham"]

@app.post("/feedback")
def collect_feedback(feedback: FeedbackRequest):
    try:
        feedback_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": feedback.message,
            "correct_label": feedback.correct_label
        }

        # Append to a CSV file
        with open("feedback.csv", mode="a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(feedback_data)

        return {"status": "✅ Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
