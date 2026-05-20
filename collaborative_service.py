from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'bookverse_db')

collaborative_app = FastAPI(title="Book Collaborative Recommendation API")

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

@collaborative_app.get("/recommend")
async def collaborative_recommend(
    user_id: int = Query(..., description="User ID"),
    top_n: int = Query(10, description="Number of recommendations", ge=1, le=50)
):
    start_time = time.time()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user's cart items (A + B)
        cursor.execute("""
            SELECT DISTINCT b.id, b.title, b.author
            FROM carts c 
            INNER JOIN cart_items ci ON c.id = ci.cart_id 
            INNER JOIN books b ON b.id = ci.book_id
            WHERE c.user_id = %s
        """, [user_id])
        cart_items = cursor.fetchall()
        
        if not cart_items:
            return {
                "user_id": user_id,
                "message": "No items in cart",
                "recommendations": []
            }
        
        # Get all transactions for collaborative filtering
        cursor.execute("""
            SELECT 
                t.id,
                (SELECT GROUP_CONCAT(b.id SEPARATOR ',')
                 FROM transaction_items ti
                 INNER JOIN books b ON ti.book_id = b.id
                 WHERE ti.transaction_id = t.id) as items 
            FROM transactions t WHERE t.user_id != %s AND t.status = 'done'
        """, [user_id])
        transactions = cursor.fetchall()
        
        # Build item co-occurrence vectors for cosine similarity
        all_items = set()
        item_vectors = defaultdict(lambda: defaultdict(int))
        
        # Collect all unique items and build co-occurrence matrix
        for trans in transactions:
            if trans[1]:
                items = [int(x) for x in trans[1].split(',')]
                all_items.update(items)
                
                # Build vectors: each item is represented by what it co-occurs with
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        item_a, item_b = items[i], items[j]
                        item_vectors[item_a][item_b] += 1
                        item_vectors[item_b][item_a] += 1
        
        if not all_items:
            return {
                "user_id": user_id,
                "message": "No transaction data available",
                "recommendations": []
            }
        
        # Create item-to-index mapping
        item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        n_items = len(all_items)
        
        # Build item matrix for cosine similarity
        item_matrix = np.zeros((n_items, n_items))
        for item, co_occurrences in item_vectors.items():
            idx = item_to_idx[item]
            for other_item, count in co_occurrences.items():
                other_idx = item_to_idx[other_item]
                item_matrix[idx, other_idx] = count
        
        # Calculate cosine similarity matrix
        # Normalize rows to unit vectors
        row_norms = np.sqrt(np.sum(item_matrix ** 2, axis=1))
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        normalized_matrix = item_matrix / row_norms[:, np.newaxis]
        
        # Cosine similarity matrix
        cosine_sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        
        # Get cart item IDs and their indices
        cart_item_ids = [item[0] for item in cart_items]
        cart_item_indices = [item_to_idx[item_id] for item_id in cart_item_ids if item_id in item_to_idx]
        
        if not cart_item_indices:
            return {
                "user_id": user_id,
                "cart_items": [{"id": item[0], "title": item[1]} for item in cart_items],
                "message": "Cart items not found in transaction history",
                "recommendations": []
            }
        
        # Create user profile by averaging cart item vectors
        user_profile = np.zeros(n_items)
        for idx in cart_item_indices:
            user_profile += cosine_sim_matrix[idx]
        user_profile = user_profile / len(cart_item_indices)
        
        # Get recommendations (items not in cart)
        recommendations = {}
        for idx, score in enumerate(user_profile):
            item_id = idx_to_item[idx]
            if item_id not in cart_item_ids and score > 0:
                recommendations[item_id] = score
        
        # Sort by similarity score
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not sorted_recs:
            return {
                "user_id": user_id,
                "cart_items": [{"id": item[0], "title": item[1]} for item in cart_items],
                "message": "No recommendations found",
                "recommendations": []
            }
        
        # Get book details for recommendations
        placeholders = ','.join(['%s'] * len(sorted_recs))
        cursor.execute(f"""
            SELECT id, title, author, 
                   COALESCE(description, '') as description,
                   COALESCE(series, '') as series
            FROM books 
            WHERE id IN ({placeholders})
        """, [rec[0] for rec in sorted_recs])
        
        recommended_books = cursor.fetchall()
        
        # Create response
        recommendations_list = []
        for book in recommended_books:
            score = dict(sorted_recs).get(book[0], 0)
            recommendations_list.append({
                "book_id": book[0],
                "title": book[1],
                "author": book[2],
                "description": book[3][:150] + "..." if len(book[3]) > 150 else book[3],
                "similarity_score": round(score, 4)
            })
        
        # Get similar items for each cart item (for insights)
        similar_items_insights = []
        for cart_item_idx in cart_item_indices[:3]:  # Top 3 cart items
            cart_item_id = idx_to_item[cart_item_idx]
            cart_item_title = next((item[1] for item in cart_items if item[0] == cart_item_id), "Unknown")
            
            similarities = cosine_sim_matrix[cart_item_idx]
            top_similar_idx = similarities.argsort()[-6:-1][::-1]  # Top 5 similar items
            
            similar_books = []
            for sim_idx in top_similar_idx:
                sim_item_id = idx_to_item[sim_idx]
                if sim_item_id not in cart_item_ids and similarities[sim_idx] > 0:
                    similar_books.append({
                        "book_id": sim_item_id,
                        "similarity": round(float(similarities[sim_idx]), 4)
                    })
            
            if similar_books:
                similar_items_insights.append({
                    "cart_item": {
                        "id": cart_item_id,
                        "title": cart_item_title
                    },
                    "similar_books": similar_books[:3]
                })
        
        return {
            "user_id": user_id,
            "cart_items": [{"id": item[0], "title": item[1]} for item in cart_items],
            "recommendations": recommendations_list,
            "recommendation_method": "Item-based Collaborative Filtering using Cosine Similarity",
            "cosine_similarity_insights": similar_items_insights,
            "stats": {
                "total_items_analyzed": n_items,
                "total_transactions_analyzed": len(transactions),
                "cart_items_count": len(cart_items)
            },
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()
