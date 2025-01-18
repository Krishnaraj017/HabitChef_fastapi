import os
from fastapi import FastAPI
from app.routes import prediction_router
import uvicorn

app = FastAPI(title="Food Prediction API")

# Include routes
app.include_router(prediction_router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Food Prediction API"}

if __name__ == "__main__":
    # Use the PORT environment variable or default to 10000
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
