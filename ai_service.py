from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the pre-trained model for text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Create a Pydantic model for request validation
class TextRequest(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-powered API!"}

# Endpoint for text analysis (POST request)
@app.post("/analyze/")
def analyze_text(request: TextRequest):
    result = classifier(request.text)
    return {"label": result[0]['label'], "score": result[0]['score']}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

