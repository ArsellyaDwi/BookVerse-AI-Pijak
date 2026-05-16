from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'bookverse_db')

# Create sub-app for book recommendations
book_app = FastAPI(title="Book Recommendation API")

book_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )

def combine_text_features(row):
    return f"{row['title']} {row['series']} {row['description']} {row['author']} {row['genres']}"

@book_app.get("/recommend/{user_id}")
async def recommend(
    user_id: int, 
    book_id: int = None,
    top_n: int = 10
):
    start_time = time.time()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id, email FROM users WHERE id = %s", [user_id])
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        current_book_that_user_lokking = []

        if book_id != None:
            cursor.execute("""
                SELECT 
                    id, 
                    title, 
                    COALESCE(series, '') as series, 
                    COALESCE(description, '') as description, 
                    COALESCE(author, '') as author,
                    (SELECT COALESCE(GROUP_CONCAT(g.name SEPARATOR ' '), '')
                    FROM book_genre bg
                    INNER JOIN genres g ON bg.genre_id = g.id
                    WHERE bg.book_id = id) as genres
                FROM books
                WHERE id = %s
            """, [book_id])
            current_book_that_user_lokking = cursor.fetchall()


        cursor.execute("""
            SELECT 
                wi.book_id as id, 
                b.title, 
                COALESCE(b.series, '') as series, 
                COALESCE(b.description, '') as description, 
                COALESCE(b.author, '') as author,
                (SELECT COALESCE(GROUP_CONCAT(g.name SEPARATOR ' '), '')
                 FROM book_genre bg
                 INNER JOIN genres g ON bg.genre_id = g.id
                 WHERE bg.book_id = b.id) as genres
            FROM wishlists w 
            INNER JOIN wishlist_items wi ON w.id = wi.wishlist_id 
            INNER JOIN books b ON b.id = wi.book_id
            WHERE w.user_id = %s
        """, [user_id])
        wishlists = cursor.fetchall()
        
        cursor.execute("""
            SELECT 
                ci.book_id as id, 
                b.title, 
                COALESCE(b.series, '') as series, 
                COALESCE(b.description, '') as description, 
                COALESCE(b.author, '') as author,
                (SELECT COALESCE(GROUP_CONCAT(g.name SEPARATOR ' '), '')
                 FROM book_genre bg
                 INNER JOIN genres g ON bg.genre_id = g.id
                 WHERE bg.book_id = b.id) as genres
            FROM carts c 
            INNER JOIN cart_items ci ON c.id = ci.cart_id 
            INNER JOIN books b ON b.id = ci.book_id
            WHERE c.user_id = %s
        """, [user_id])
        carts = cursor.fetchall()
        
        seen_ids = set()
        user_books = []
        for book in wishlists + carts + current_book_that_user_lokking:
            if book[0] not in seen_ids:
                seen_ids.add(book[0])
                user_books.append(book)
        
        if not user_books:
            return {
                "user_id": user_id,
                "user_email": user[1],
                "total_user_books": 0,
                "recommendations": [],
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        placeholders = ','.join(['%s'] * len(user_books))
        cursor.execute(f"""
            SELECT 
                b.id, b.title, 
                COALESCE(b.series, '') as series,
                COALESCE(b.description, '') as description,
                COALESCE(b.author, '') as author,
                (SELECT COALESCE(GROUP_CONCAT(g.name SEPARATOR ' '), '')
                 FROM book_genre bg
                 INNER JOIN genres g ON bg.genre_id = g.id
                 WHERE bg.book_id = b.id) as genres
            FROM books b
            WHERE b.id NOT IN ({placeholders})
        """, [book[0] for book in user_books])
        other_books = cursor.fetchall()
        
        if not other_books:
            return {
                "user_id": user_id,
                "user_email": user[1],
                "total_user_books": len(user_books),
                "recommendations": [],
                "message": "No other books available for recommendation",
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        user_df = pd.DataFrame(user_books, columns=['id', 'title', 'series', 'description', 'author', 'genres'])
        other_df = pd.DataFrame(other_books, columns=['id', 'title', 'series', 'description', 'author', 'genres'])
        
        user_df['text'] = user_df.apply(combine_text_features, axis=1)
        other_df['text'] = other_df.apply(combine_text_features, axis=1)
        
        all_text = pd.concat([user_df['text'], other_df['text']], ignore_index=True)
        
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000, 
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(all_text)
        
        user_tfidf = tfidf_matrix[:len(user_df)]
        other_tfidf = tfidf_matrix[len(user_df):]
        
        user_tfidf_dense = user_tfidf.toarray()
        other_tfidf_dense = other_tfidf.toarray()
        
        user_profile = user_tfidf_dense.mean(axis=0).reshape(1, -1)
        
        similarities = cosine_similarity(user_profile, other_tfidf_dense).flatten()
        
        top_n = min(top_n, len(similarities))
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:
                book = other_df.iloc[idx]
                recommendations.append({
                    "book_id": int(book['id']),
                    "title": book['title'],
                    "author": book['author'],
                    "genres": book['genres'],
                    "description": book['description'][:200] + "..." if len(book['description']) > 200 else book['description'],
                    "similarity_score": round(float(similarities[idx]), 4)
                })

        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            "user_id": user_id,
            "user_email": user[1],
            "total_user_books": len(user_books),
            "user_books": [{"id": b[0], "title": b[1]} for b in user_books[:5]],
            "recommendations": recommendations,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@book_app.post("/recommend/")
async def recommend_post(
    user_id: int, 
    top_n: int = 10,
    book_id: int = None
):
    return await recommend(user_id, book_id, top_n)