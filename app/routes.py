from fastapi import APIRouter, UploadFile, HTTPException
import os
from app.services import predict_food

prediction_router = APIRouter()


@prediction_router.post("/predict")
async def predict(file: UploadFile):
    """
    Endpoint to predict food from an image.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only JPEG and PNG are supported."
        )

    # Define the directory and file path
    directory = "temp"
    image_path = os.path.join(directory, file.filename)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the uploaded file
    try:
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # Call the prediction service
        predictions = predict_food(image_path)
        return {"predicted_foods": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup the temp file
        if os.path.exists(image_path):
            os.remove(image_path)
