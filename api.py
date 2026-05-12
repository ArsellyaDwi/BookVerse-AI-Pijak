from fastapi import FastAPI, Header, HTTPException, UploadFile, File
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib
import mysql.connector
import time
import csv
import io
from pydantic import BaseModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendRequest(BaseModel):
    text: str

EMOTION_RULES = {
    "anger": {
        "emotion": ["relief", "happiness"],
        "genre": ["fiction"]
    },
    "sadness": {
        "emotion": ["hope", "love"],
        "genre": ["self-help", "motivational"]
    },
    "anxiety": {
        "emotion": ["relief"],
        "genre": ["self-help"]
    },
    "loneliness": {
        "emotion": ["love", "gratitude"],
        "genre": ["romance"]
    },
    "fear": {
        "emotion": ["hope"],
        "genre": ["motivational"]
    }
}

API_KEY = "RAHASIA"
BASE_MODEL_PATH = "./base_model/distilbert-base-uncased"
TRAINED_MODEL_PATH = "./trained_model/model.pth"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "ai_exp"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

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

class AIController:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        self.mlb = joblib.load("label_encoder.pkl")
        self.model = EmotionClassifier(BASE_MODEL_PATH, len(self.mlb.classes_))
        self.model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
        self.model.to(device)
        self.model.eval()

    def verify_api_key(self, x_api_key):
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API KEY")

    def get_db(self):
        return mysql.connector.connect(**DB_CONFIG)

    def apply_rules(self, emotions):
        boosted_emotions = set(emotions)
        genres = set()
        for emo in emotions:
            if emo in EMOTION_RULES:
                rule = EMOTION_RULES[emo]
                boosted_emotions.update(rule.get("emotion", []))
                genres.update(rule.get("genre", []))
        return list(boosted_emotions), list(genres)

    def predict(self, text, threshold=0.5):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        emotions = []
        for i, p in enumerate(probs):
            if p >= threshold:
                emotions.append(self.mlb.classes_[i])
        return emotions

    def get_dataset(self):
        db = self.get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM ai_emotion_datasets")
        data = cursor.fetchall()
        return {"data": data}

    def recommend(self, text):
        emotions = self.predict(text)
        boosted_emotions, genres = self.apply_rules(emotions)
        db = self.get_db()
        cursor = db.cursor(dictionary=True)
        query = """
        SELECT b.id, b.mood_tags, GROUP_CONCAT(g.name) as genres
        FROM books b
        LEFT JOIN book_genres bg ON b.id = bg.book_id
        LEFT JOIN genres g ON bg.genre_id = g.id
        GROUP BY b.id
        """
        cursor.execute(query)
        books = cursor.fetchall()
        if not books:
            return {"book_ids": []}
        book_ids = []
        corpus = []
        for b in books:
            book_ids.append(b["id"])
            tag_text = b["mood_tags"] if b["mood_tags"] else ""
            genre_text = b["genres"] if b["genres"] else ""
            combined = f"{tag_text} {genre_text}"
            corpus.append(combined)
        user_text = " ".join(boosted_emotions + genres)
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(corpus + [user_text])
        user_vector = tfidf_matrix[-1]
        book_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(user_vector, book_vectors)[0]
        scored = list(zip(book_ids, similarities))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_books = [bid for bid, score in scored if score > 0]
        return {"book_ids": top_books[:10]}

    def train(self):
        start_time = time.time()
        db = self.get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT text, labels FROM ai_emotion_datasets")
        data = cursor.fetchall()
        texts = []
        labels = []
        for row in data:
            texts.append(row["text"])
            labels.append(row["labels"].split("|"))
        mlb_local = MultiLabelBinarizer()
        y = mlb_local.fit_transform(labels)
        joblib.dump(mlb_local, "label_encoder.pkl")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        total_loss = 0
        for epoch in range(2):
            for text, label in zip(texts, y):
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                inputs.pop("token_type_ids", None)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                label = torch.tensor(label, dtype=torch.float).unsqueeze(0).to(device)
                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        torch.save(self.model.state_dict(), TRAINED_MODEL_PATH)
        end_time = time.time()
        return {
            "loss": round(total_loss, 4),
            "training_time_seconds": round(end_time - start_time, 2)
        }

    def tag_all_books(self):
        db = self.get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, description FROM books")
        books = cursor.fetchall()
        updated = []
        for book in books:
            text = book["description"] or ""
            emotions = self.predict(text)
            tags = "|".join(emotions)
            cursor.execute(
                "UPDATE books SET mood_tags=%s WHERE id=%s",
                (tags, book["id"])
            )
            updated.append({
                "book_id": book["id"],
                "tags": emotions
            })
        db.commit()
        return {"updated_books": updated}

    def import_emotions_csv(self, file: UploadFile):
        db = self.get_db()
        cursor = db.cursor()
        content = file.file.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            cursor.execute(
                "INSERT INTO ai_emotion_datasets (text, labels) VALUES (%s, %s)",
                (row["text"], row["labels"])
            )
        db.commit()
        return {"status": "success"}

    def import_books_csv(self, file: UploadFile):
        db = self.get_db()
        cursor = db.cursor()
        content = file.file.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            cursor.execute(
                "INSERT INTO books (title, description) VALUES (%s, %s)",
                (row["title"], row["description"])
            )
            book_id = cursor.lastrowid
            genres = row["genres"].split("|") if row.get("genres") else []
            for g_name in genres:
                cursor.execute("SELECT id FROM genres WHERE name=%s", (g_name,))
                g = cursor.fetchone()
                if g:
                    genre_id = g[0]
                else:
                    cursor.execute("INSERT INTO genres (name) VALUES (%s)", (g_name,))
                    genre_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO book_genres (book_id, genre_id) VALUES (%s, %s)",
                    (book_id, genre_id)
                )
        db.commit()
        return {"status": "success"}

controller = AIController()

@app.get("/dataset")
def get_dataset(x_api_key: str = Header(...)):
    controller.verify_api_key(x_api_key)
    return controller.get_dataset()

@app.post("/recommend")
def recommend(req: RecommendRequest, x_api_key: str = Header(...)):
    controller.verify_api_key(x_api_key)
    return controller.recommend(req.text)

@app.post("/train")
def train(x_api_key: str = Header(...)):
    controller.verify_api_key(x_api_key)
    return controller.train()

@app.post("/tag-book")
def tag_books(x_api_key: str = Header(...)):
    controller.verify_api_key(x_api_key)
    return controller.tag_all_books()

@app.post("/import-emotions-csv")
def import_emotions(file: UploadFile = File(...), x_api_key: str = Header(...)):
    controller.verify_api_key(x_api_key)
    return controller.import_emotions_csv(file)

@app.post("/import-books-csv")
def import_books(file: UploadFile = File(...), x_api_key: str = Header(...)):
    controller.verify_api_key(x_api_key)
    return controller.import_books_csv(file)
