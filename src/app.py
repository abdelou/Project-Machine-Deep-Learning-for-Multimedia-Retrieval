from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from feature_extractor import ViT_Extractor, ResNet34_ImageNet_Extractor
from multimodal_retrieval import MultimodalEngine
from retrieval import getkVoisins
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize engines
# Note: In a real scenario, we would load pre-indexed features
unimodal_extractor = ViT_Extractor()
multimodal_engine = MultimodalEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_unimodal', methods=['POST'])
def search_unimodal():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    
    # Simple example: search using ViT
    # This assumes features are already indexed in some directory
    # For demonstration, we just return the query for now or mock it
    return jsonify({"message": "Unimodal search triggered (ViT)", "query": img_path})

@app.route('/search_multimodal', methods=['POST'])
def search_multimodal():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Text-to-Image search
    results = multimodal_engine.search_text_to_image(query, k=10)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
