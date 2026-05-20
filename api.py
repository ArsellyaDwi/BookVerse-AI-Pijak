from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from emotion_service import emotion_app
from content_based_service import book_app
from personality_service import personality_app
from collaborative_service import collaborative_app

load_dotenv()

API_KEY = os.getenv('API_KEY', 'rahasia')
API_KEY_NAME = os.getenv('API_KEY_NAME', 'X-API-Key')

app = FastAPI(title="Combined API - BookVerse & Emotion Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    api_key = request.headers.get(API_KEY_NAME)
    
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "MISSING_API_KEY",
                "message": f"Missing API Key. Please provide '{API_KEY_NAME}' header",
                "required_header": API_KEY_NAME
            }
        )
    
    if api_key != API_KEY:
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": "INVALID_API_KEY",
                "message": "Invalid API Key. Please check your API key and try again",
                "provided_key": api_key[:10] + "..." if len(api_key) > 10 else api_key
            }
        )
    
    return await call_next(request)

app.mount("/api/content-based", book_app)
app.mount("/api/emotion", emotion_app)
app.mount("/api/personality", personality_app)
app.mount("/api/collaborative", collaborative_app)

@app.get("/")
async def root():
    return {
        "message": "Combined API Services",
        "authentication": "API Key required for ALL endpoints",
        "api_key_header": API_KEY_NAME
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=False)