ffrom fastapi import FastAPI, UploadFile, File, Header, HTTPException

app = FastAPI()

API_KEY = "MY_SECRET_KEY_123"

@app.post("/api/detect")  # Route will be /api/detect
async def detect_audio(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {
        "status": "success",
        "filename": file.filename
    }
