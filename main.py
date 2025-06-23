from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Load model and vectorizer
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "vectorizer.joblib"))

# Create FastAPI app
app = FastAPI(title="Job Scam Detector API")

# Request schema
class JobPost(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "🚀 Job Scam Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(post: JobPost):
    # Combine input fields into a single text
    full_text = f"{post.title} {post.company_profile} {post.description} {post.requirements}"
    vect_text = vectorizer.transform([full_text])
    
    prediction = model.predict(vect_text)[0]
    confidence = model.predict_proba(vect_text)[0][1]  # Probability of being scam

    return {
        "prediction": int(prediction),
        "confidence": round(confidence, 2),
        "message": "🚨 Scam Detected!" if prediction == 1 else "✅ Looks Legit!"
    }
