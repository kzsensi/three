import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import Cores
from core.face_model import FaceEmotionModel
from core.speech_model import SpeechEmotionModel
from core.text_model import TextEmotionModel
from core.fusion import AdaptiveAttentionFusion, FusionEngine

# Import API specific
from api.workers import WorkerPool
from api.websockets import router as ws_router, init_websockets

app = FastAPI(title="Multimodal Emotion Recognition API")

# Add standard CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
worker_pool = None

@app.on_event("startup")
async def startup_event():
    global worker_pool
    
    print("Initializing Modality Models (Loading checkpoints)...")
    face_model = FaceEmotionModel()
    speech_model = SpeechEmotionModel()
    text_model = TextEmotionModel()
    fusion_engine = FusionEngine()
    
    print("Wiring up Worker Pool...")
    worker_pool = WorkerPool(face_model, speech_model, text_model, fusion_engine)
    init_websockets(worker_pool)
    
    # Spin up asyncio tasks for the workers
    asyncio.create_task(worker_pool.face_worker())
    asyncio.create_task(worker_pool.audio_worker())
    
    print("Application Ready and Models Loaded.")

app.include_router(ws_router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Multimodal System is active"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
