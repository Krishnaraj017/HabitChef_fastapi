import os
from fastapi import FastAPI
from app.routes import prediction_router
import uvicorn

app = FastAPI(title="Food Prediction API")

# Include routes
app.include_router(prediction_router)


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
