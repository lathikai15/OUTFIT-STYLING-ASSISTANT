from flask import Flask, request, render_template, jsonify
import os
from image_processor import ImageProcessor

app = Flask(__name__)

# Configuration
COLOR_CSV_PATH = r'C:\Users\lathi\OneDrive\Desktop\OUTFIT_AI\colors.csv'
MODEL_NAME = "facebook/deit-base-patch16-224"
API_KEY = "AIzaSyBzBDoQltlwEvJx98Y2IDEMlL1ZuWGp8D0"  # Replace with your actual API key

# Initialize the image processor
image_processor = ImageProcessor(COLOR_CSV_PATH, MODEL_NAME, API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    img_path = os.path.join("uploaded_images", file.filename)
    file.save(img_path)

    # Process the image and generate outfit suggestion
    results = image_processor.process_image(img_path)
    if results is None:
        return jsonify({"error": "Could not process image."}), 500

    suggestion = image_processor.generate_outfit_suggestion(results)
    
    return jsonify({
        "results": results,
        "suggestion": suggestion
    })

if __name__ == '__main__':
    app.run(debug=True)
