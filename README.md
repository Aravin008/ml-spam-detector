# Spam Detector (FastAPI + Scikit-learn)

A simple machine learning spam classifier built using Python, scikit-learn, and FastAPI.

This project trains a text classification model to detect whether an SMS message is spam or ham (not spam) using the classic [SMS Spam Collection Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv).  
The trained model is then served through a FastAPI backend for real-time predictions.

---

## Features

- TF-IDF + Logistic Regression for spam detection
- FastAPI-based REST API for inference
- Model and vectorizer persistence using joblib
- Feedback API for collecting user corrections
- Docker support for easy deployment

---

## Project Structure

```
.
├── app.py
├── train_model.py
├── models/
├── feedback.csv
├── requirements.txt
├── Dockerfile.dev
└── docker-compose.yml
````

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Aravind008/ml-spam-detector.git
cd ml-spam-detector
````

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This will:

* Download and load the dataset
* Train the model
* Save files under `models/spam_model.pkl` and `models/vectorizer.pkl`

---

## Run the API

### Option 1: Local

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### Option 2: Docker

```bash
docker compose up --build
```

---

## API Endpoints

### 1. Predict Spam

**POST** `/predict`

Request:

```json
{
  "message": "Congratulations! You've won a free ticket!"
}
```

Response:

```json
{
  "message": "Congratulations! You've won a free ticket!",
  "prediction": "spam"
}
```

### 2. Submit Feedback

**POST** `/feedback`

Request:

```json
{
  "message": "Hello, how are you?",
  "correct_label": "ham"
}
```

Response:

```json
{
  "status": "Feedback recorded"
}
```

Feedback is appended to `feedback.csv` for later use in retraining.
