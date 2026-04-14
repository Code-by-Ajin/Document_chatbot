from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

app = FastAPI(title="Document Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files to serve the frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the RAG engine
engine = RAGEngine()

# Initialize MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
if MONGODB_URI:
    try:
        mongo_client = MongoClient(MONGODB_URI)
        db = mongo_client["document_chatbot"]
        chats_collection = db["chats"]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        chats_collection = None
else:
    chats_collection = None

MAX_CHATS = 50

class QuestionRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        engine.ingest_document(file_path)
        return {"success": True, "message": f"Successfully processed {file.filename}. You can now start asking questions."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        answer = engine.ask_question(request.question)
        
        # Save to MongoDB and enforce limit
        if chats_collection is not None:
            chats_collection.insert_one({
                "question": request.question,
                "answer": answer
            })
            
            msg_count = chats_collection.count_documents({})
            if msg_count > MAX_CHATS:
                oldest_cursor = chats_collection.find().sort("_id", 1).limit(5)
                oldest_ids = [doc["_id"] for doc in oldest_cursor]
                chats_collection.delete_many({"_id": {"$in": oldest_ids}})

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    if chats_collection is None:
        return {"messages": []}
    
    docs = chats_collection.find().sort("_id", 1)
    messages = []
    for doc in docs:
        messages.append({
            "question": doc["question"],
            "answer": doc["answer"]
        })
    return {"messages": messages}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
