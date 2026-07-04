from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import tempfile
import os

# Reduce TensorFlow verbose logs when modules import TF. Set before importing `predict`.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from predict import predict_video, get_model
from utils import preload_landmarker_assets

app = FastAPI(
    title="Sign-to-Text API",
    description="Convert sign language videos to text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


@app.on_event("startup")
async def warm_up_model():
    await run_in_threadpool(get_model)
    await run_in_threadpool(preload_landmarker_assets)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = await run_in_threadpool(predict_video, tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return JSONResponse(content=result)


if __name__ == "__main__":
    # Allow running the app directly for convenience. Prefer `uvicorn main:app` for production.
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
