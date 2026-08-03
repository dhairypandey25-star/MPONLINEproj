import cv2
import numpy as np
from fastapi import UploadFile
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from typing import Tuple, Optional

_cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
_vision_model = MobileNetV2(weights='imagenet')

async def _extract_image_array(upload: UploadFile) -> np.ndarray:
    """Asynchronously reads an uploaded image and decodes it into an OpenCV array."""
    file_bytes = await upload.read()
    numpy_buffer = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(numpy_buffer, cv2.IMREAD_COLOR)

async def identify_customer(upload: UploadFile) -> Tuple[Optional[str], str, float]:
    """Detects a face in the image and attempts to match it against registered customer profiles."""
    image_frame = await _extract_image_array(upload)
    grayscale_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    
    detected_faces = _cascade_classifier.detectMultiScale(
        grayscale_frame, scaleFactor=1.1, minNeighbors=4
    )
    
    if len(detected_faces) == 0:
        return None, "UNKNOWN", 0.0
        
    (x_coord, y_coord, width, height) = detected_faces[0]
    face_region = grayscale_frame[y_coord:y_coord + height, x_coord:x_coord + width]
    
    try:
        predicted_label, match_distance = _lbph_recognizer.predict(face_region)
        
        if match_distance < 70:
            confidence_score = float(100 - match_distance) / 100.0
            return f"Customer_{predicted_label}", "REGISTERED", confidence_score
            
        return None, "UNKNOWN", 0.0
    except cv2.error:
        return None, "UNKNOWN", 0.0

async def categorize_product(upload: UploadFile) -> Tuple[str, float]:
    """Processes an image through MobileNetV2 to determine the product category."""
    image_frame = await _extract_image_array(upload)
    
    resized_image = cv2.resize(image_frame, (224, 224))
    expanded_image = np.expand_dims(resized_image, axis=0)
    processed_image = preprocess_input(expanded_image)
    
    raw_predictions = _vision_model.predict(processed_image)
    top_prediction = decode_predictions(raw_predictions, top=1)[0][0]
    
    return top_prediction[1], float(top_prediction[2])
