from collections import defaultdict
from ultralytics import YOLO
from app.utils import label_mapping


def predict_food(image_path):
    """
    Predict food using multiple YOLO models and return the highest-confidence predictions.
    """
    predictions = defaultdict(float)
    model_paths = ["models/best (1).pt", "models/best(2).pt"]

    for model_path in model_paths:
        model = YOLO(model_path)
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
