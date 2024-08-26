from flask import Flask, request, jsonify
import subprocess
import json
import os

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
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

        # Run the YOLO command
        command = f"yolo detect predict model=yolov8n.pt source={image_path} --save-txt --save-conf --format json"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            return jsonify({'error': 'YOLO command failed', 'details': result.stderr}), 500

        # Read the JSON output
        with open('runs/detect/predict/predictions.json', 'r') as f:
            predictions = json.load(f)

        # Extract detected objects
        detected_objects = []
        for pred in predictions:
            for obj in pred['predictions']:
                detected_objects.append({
                    'class': obj['class'],
                    'confidence': obj['confidence'],
                    'bbox': obj['box']
                })

        # Clean up temporary files
        os.remove(image_path)
        os.system('rm -rf runs')

        return jsonify(detected_objects)

if __name__ == '__main__':
    app.run(debug=True)