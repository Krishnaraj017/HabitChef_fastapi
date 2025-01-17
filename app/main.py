from fastapi import FastAPI
from app.routes import prediction_router
app = FastAPI(title="Food Prediction API")

# Include routes
app.include_router(prediction_router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Food Prediction API"}
    