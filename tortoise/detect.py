from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import os

app = FastAPI()

# Read API key from Vercel environment variable
API_KEY = os.getenv("API_KEY")

@app.post("/detect")
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
