# ─── SETUP INSTRUCTIONS ───────────────────────────────────────
# 1. pip install -r requirements.txt
# 2. Place all 4 .pkl files in /models/
# 3. Place all 9 .png files in /static/images/
# 4. python main.py   (or: uvicorn main:app --reload --port 8000)
# 5. API docs: http://localhost:8000/docs
# 6. Run tests: pytest test_api.py -v
# ──────────────────────────────────────────────────────────────
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routes.predict import router as predict_router
from routes.metrics import router as metrics_router
from utils.model_loader import model_store, VALID_MODEL_NAMES
from utils.logger import system_logger, error_logger

app = FastAPI(
    title=os.getenv("APP_NAME", "MatPredict API"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="Material Property Prediction API — Concrete, Steel, Materials",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS: allow_credentials=True is incompatible with allow_origins=["*"] per browser spec.
# We use wildcard origin with credentials=False for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
system_logger.info("CORS configured: allow_origins=[*], allow_credentials=False")

static_path = Path(os.getenv("STATIC_DIR", "static"))
static_path.mkdir(parents=True, exist_ok=True)
images_path = static_path / "images"
images_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
system_logger.info(f"Static files served from {static_path}")

app.include_router(predict_router)
app.include_router(metrics_router)

@app.get("/health")
def health_check():
    status = "healthy"
    system_logger.info("Health check endpoint called")
    return {
        "status": status,
        "app": os.getenv("APP_NAME", "MatPredict API"),
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "models_loaded": model_store.get_loaded_models(),
        "total_models": len(VALID_MODEL_NAMES),
        "loaded_count": len(model_store.get_loaded_models())
    }

@app.get("/")
def root():
    return {"message": "MatPredict API is running", "docs": "/docs"}

@app.on_event("startup")
def startup_event():
    system_logger.info("=== MatPredict API Starting Up ===")
    system_logger.info(f"Models loaded: {model_store.get_loaded_models()}")
    system_logger.info("=== Startup complete ===")

@app.on_event("shutdown")
def shutdown_event():
    system_logger.info("=== MatPredict API Shutting Down ===")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "True") == "True",
        log_level="info"
    )

    