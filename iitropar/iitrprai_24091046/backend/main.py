import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from utils.drive_downloader import download_drive_folder
from api import predict_api, history_api

FOLDER_URL = os.getenv(
    "MODEL_FOLDER_URL",
    "https://drive.google.com/drive/folders/1oqAbBsIjLBqTCGk01CA1WvXwszcY_eLu?usp=sharing",
)
DEFAULT_LOCAL_MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "AI_ML", "models_store"))

MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_LOCAL_MODEL_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_drive_folder(FOLDER_URL, output_dir=MODEL_DIR)
    yield

app = FastAPI(title="Stock Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "model_dir": MODEL_DIR}

@app.get("/api/check-model")
def check_model():
    if not os.path.isdir(MODEL_DIR):
        raise HTTPException(status_code=404, detail=f"Model directory not found: {MODEL_DIR}")
    entries = [f.name for f in os.scandir(MODEL_DIR) if f.is_file()]
    return {"model_dir": MODEL_DIR, "files": entries}
 
app.include_router(predict_api.router, prefix="/api")
app.include_router(history_api.router, prefix="/api")
