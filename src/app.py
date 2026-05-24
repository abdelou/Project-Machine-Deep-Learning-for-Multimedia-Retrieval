from flask import Flask, render_template, request, jsonify, send_from_directory

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
unimodal_extractor = ViT_Extractor()
multimodal_engine = MultimodalEngine()
# Paths
DATA_DIR = os.path.abspath('data/dataset')
FEATURES_DIR = os.path.abspath('extracted_features')
MULTIMODAL_INDEX = os.path.join(FEATURES_DIR, 'multimodal/flickr8k_clip.index')
MULTIMODAL_PATHS = os.path.join(FEATURES_DIR, 'multimodal/flickr8k_paths.npy')

# Load Multimodal Index if exists
if os.path.exists(MULTIMODAL_INDEX):
    multimodal_engine.load_index(MULTIMODAL_INDEX, MULTIMODAL_PATHS)


# Serve images from data/dataset
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(DATA_DIR, filename)


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
    
    # 1. Extract feature from query image
    query_feature = unimodal_extractor.extract_feature(img_path)
    
    # 2. Load indexed features (ensure they are indexed first)
    # If not indexed, this might fail or be empty. 
    # For a robust app, we'd ensure indexing at start.
    if not os.path.exists(os.path.join(FEATURES_DIR, unimodal_extractor.name)):
        # Optional: Auto-index if few images, or return error
        files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            return jsonify({"error": "Database is empty. Please add images to data/dataset/"}), 404
        unimodal_extractor.index_database(files, FEATURES_DIR)

    database_features = unimodal_extractor.load_features(FEATURES_DIR)
    
    # 3. Get Top K Neighbors
    voisins = getkVoisins(database_features, query_feature, k=16, distanceName="Euclidienne")
    
    # 4. Format results
    results = []
    for path, feature, dist in voisins:

        # Construct URL to served image
        rel_path = os.path.basename(path).replace('.txt', '.jpg')
        results.append({
            "path": f"/images/{rel_path}",
            "score": 1.0 / (1.0 + float(dist)) # Convert distance to a similarity score
        })

    
    return jsonify(results)


@app.route('/search_multimodal', methods=['POST'])
def search_multimodal():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if multimodal_engine.index is None:
        return jsonify({"error": "Multimodal index not loaded. Please run indexing first."}), 404

    # Text-to-Image search
    raw_results = multimodal_engine.search_text_to_image(query, k=16)
    
    # Format results for front-end
    results = []
    for res in raw_results:
        rel_path = os.path.basename(res['path'])
        results.append({
            "path": f"/images/{rel_path}",
            "score": res['score']
        })
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
