from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
from functools import wraps

app = Flask(__name__)

# Configure the secret key
SECRET_KEY = "your_secret_key_here"  # Replace with a strong, unique secret key

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key == SECRET_KEY:
            return f(*args, **kwargs)
        else:
            return jsonify({'error': 'Unauthorized'}), 401
    return decorated

@app.route('/detect', methods=['POST'])
@require_api_key
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if image_file:
        # Save the uploaded image temporarily
        image_path = 'temp_image.jpg'
        image_file.save(image_path)

        # Perform object detection
        results = model(image_path)

        # Extract detected objects
        detected_objects = []
        for result in results:
            for box in result.boxes:
                obj = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detected_objects.append(obj)

        # Remove the temporary image file
        os.remove(image_path)

        return jsonify(detected_objects)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)