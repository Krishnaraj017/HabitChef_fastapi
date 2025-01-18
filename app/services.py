from collections import defaultdict
from ultralytics import YOLO
from app.utils import label_mapping
import gdown
import os
import tempfile


def download_from_gdrive(gdrive_url):
    """
    Download a file from Google Drive and return the local path.
    
    Args:
        gdrive_url (str): Google Drive sharing URL
        
    Returns:
        str: Path to downloaded file
    """
    # Convert sharing URL to direct download URL
    file_id = gdrive_url.split('/')[5]
    direct_url = f'https://drive.google.com/uc?id={file_id}'

    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f'model_{file_id}.pt')

    # Download if not already cached
    if not os.path.exists(temp_path):
        gdown.download(direct_url, temp_path, quiet=False)

    return temp_path


def predict_food(image_path):
    """
    Predict food using multiple YOLO models and return the highest-confidence predictions.
    """
    predictions = defaultdict(float)

    model_paths = [
        "https://drive.google.com/file/d/11k7zHJ2jM4UuGbbzpEwYxdbonH2vbTjQ/view?usp=sharing",
        "https://drive.google.com/file/d/1SqD9K881AMwLziB96cmeP7tGRJgDeZon/view?usp=sharing"
    ]

    try:
        for gdrive_url in model_paths:
            # Download model from Google Drive
            local_model_path = download_from_gdrive(gdrive_url)

            # Load and run model
            model = YOLO(local_model_path)
            results = model.predict(image_path)
            result = results[0]

            for box in result.boxes:
                label_id = box.cls[0].item()
                confidence = float(box.conf.item())

                label = result.names[label_id]
                if label in label_mapping:
                    label = label_mapping[label]

                if confidence > predictions[label]:
                    predictions[label] = confidence

        return list(predictions.keys())

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
