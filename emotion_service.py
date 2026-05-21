from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, AutoModel
import joblib
import numpy as np
import os
import time
from dotenv import load_dotenv
import mysql.connector
import json

load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'bookverse_db')


def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )

# Create sub-app for emotion classification
emotion_app = FastAPI()

class EmotionRequest(BaseModel):
    text: str

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EmotionClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled)
        return self.fc(x)

try:
    label_encoder = joblib.load("./trained_model/label_encoder.pkl")
    print(f"Label encoder loaded. Classes: {label_encoder.classes_}")
    num_labels = len(label_encoder.classes_)
except FileNotFoundError:
    print("Error: label_encoder.pkl not found!")
    label_encoder = None
    num_labels = 0

BASE_MODEL_PATH = "distilbert-base-uncased"
TRAINED_MODEL_PATH = "./trained_model/model.pth"

try:
    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL_PATH)
    print("Tokenizer loaded successfully")
    
    model = EmotionClassifier(BASE_MODEL_PATH, num_labels)
    
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

@emotion_app.get("/health")
def health():
    return {
        "status": "AI service is running",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "encoder_loaded": label_encoder is not None,
        "num_labels": num_labels
    }

@emotion_app.post("/predict")
async def predict_emotion(request: EmotionRequest):
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded properly")
    
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs.pop("token_type_ids", None)
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

        top_indices = np.argsort(probabilities)[::-1][:3]
        
        results = []
        for idx in top_indices:
            if probabilities[idx] > 0.3:
                results.append({
                    "emotion": label_encoder.classes_[idx],
                    "confidence": float(probabilities[idx])
                })
        
        return {"success": True, "predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@emotion_app.post("/tag-books")
async def tag_books():
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded properly")
    
    try:
        start_time = time.time()
    
        conn = get_db_connection()
        cursor = conn.cursor()
            
        cursor.execute("""
            SELECT id, title, description FROM books 
            WHERE description IS NOT NULL AND description != ''
        """)
        
        all_books = cursor.fetchall()
        
        if not all_books:
            return {"success": True, "message": "No books found to tag", "books_processed": 0}
        
        updated_count = 0
        
        for book in all_books:
            book_id = book[0]
            title = book[1]
            description = book[2]
            
            inputs = tokenizer(
                description,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            inputs.pop("token_type_ids", None)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = model(inputs["input_ids"], inputs["attention_mask"])
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
            all_emotions = []
            for idx, prob in enumerate(probabilities):
                if prob > 0.3:
                    all_emotions.append({
                        "emotion": label_encoder.classes_[idx],
                        "confidence": float(prob)
                    })

            all_emotions.sort(key=lambda x: x["confidence"], reverse=True)
            mood_tags_json = json.dumps(all_emotions)

            cursor.execute(
                "UPDATE books SET mood_tags = %s WHERE id = %s",
                [mood_tags_json, book_id]
            )
            
            updated_count += 1
        
        conn.commit()
        
        elapsed_time = time.time() - start_time
        
        return {
            "success": True, 
            "message": f"Successfully tagged {updated_count} books",
            "books_processed": updated_count,
            "total_books_checked": len(all_books),
            "execution_time_seconds": round(elapsed_time, 2),
            "mood_tags_format": "[{emotion: string, confidence: float}]"
        }  
            
        return {"success": True, "message": "All books tagged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@emotion_app.post("/tag-books-single")
async def tag_books(book_id: int):
    if model is None or tokenizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded properly")
    
    try:
        start_time = time.time()
    
        conn = get_db_connection()
        cursor = conn.cursor()
            
        cursor.execute("""
            SELECT id, title, description FROM books 
            WHERE description IS NOT NULL AND description != '' AND id = %s
        """, [book_id])
        
        book = cursor.fetchone()
        
        if not book:
            return {"success": True, "message": "No books found to tag", "books_processed": 0}
        
        updated_count = 0
        
        book_id = book[0]
        title = book[1]
        description = book[2]
        
        inputs = tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        all_emotions = []
        for idx, prob in enumerate(probabilities):
            if prob > 0.3:
                all_emotions.append({
                    "emotion": label_encoder.classes_[idx],
                    "confidence": float(prob)
                })

        all_emotions.sort(key=lambda x: x["confidence"], reverse=True)
        mood_tags_json = json.dumps(all_emotions)

        cursor.execute(
            "UPDATE books SET mood_tags = %s WHERE id = %s",
            [mood_tags_json, book_id]
        )
        
        updated_count += 1
        
        conn.commit()
        
        elapsed_time = time.time() - start_time
        
        return {
            "success": True, 
            "message": f"Successfully tagged {updated_count} books",
            "books_processed": updated_count,
            "total_books_checked": 1,
            "execution_time_seconds": round(elapsed_time, 2),
            "mood_tags_format": "[{emotion: string, confidence: float}]"
        }  
            
        return {"success": True, "message": "All books tagged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@emotion_app.get("/emotions")
def get_emotions():
    if label_encoder is None:
        return {"emotions": ["happiness", "sadness", "anxiety", "fear", "love", "relief"]}
    return {"emotions": label_encoder.classes_.tolist()}