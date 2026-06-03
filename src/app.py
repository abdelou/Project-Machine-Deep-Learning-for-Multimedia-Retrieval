from flask import Flask, render_template, request, jsonify, send_from_directory

import os
import cv2
import numpy as np
from feature_extractor import *
from multimodal_retrieval import MultimodalEngine
from retrieval import getkVoisins, extract_class_id
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize engines
extractors = {
    "ViT": ViT_Extractor(),
    "BGR": Histogram_Color_Extractor(),
    "HSV": Histogram_HSV_Extractor(),
    "SIFT": SIFT_Extractor(),
    "ORB": ORB_Extractor(),
    "LBP": LBP_Extractor(),
    "GLCM": GLCM_Extractor(),
    "HOG": HOG_Extractor()
}
multimodal_engine = MultimodalEngine()

def create_plot(k, retrieved_classes, query_classes, total_relevant):
    precisions = []
    recalls = []
    relevant_count = 0
    query_class = query_classes[0] if isinstance(query_classes, list) else query_classes

    for i in range(k):
        if i < len(retrieved_classes) and retrieved_classes[i] == query_class:
            relevant_count += 1
        precisions.append(relevant_count / (i + 1))
        recalls.append(relevant_count / total_relevant if total_relevant > 0 else 0)
    
    plt.figure(figsize=(9, 7))
    plt.plot(recalls, precisions, marker='.', color='#55b6a3')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall-Precision Curve')
    plt.ylim([0, 1.05])
    plt.xlim([0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=100)
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    return img_base64

# Paths relative to this script
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'dataset')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'extracted_features')

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
    
    descriptor_name = request.form.get('descriptor', 'ViT')
    distance_name = request.form.get('distance', 'Euclidienne')
    extractor = extractors.get(descriptor_name, extractors['ViT'])
    
    # Auto-adjust distance for SIFT/ORB if user forgot
    if descriptor_name in ["SIFT", "ORB"] and distance_name not in ["Brute force", "Flann"]:
        distance_name = "Brute force"
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for unimodal search. Please load an image first."}), 400
        
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    
    # 1. Extract feature from query image
    query_feature = extractor.extract_feature(img_path)
    
    # 2. Indexing check (simplified for web)
    desc_features_dir = os.path.join(FEATURES_DIR, extractor.name)
    if not os.path.exists(os.path.join(desc_features_dir, "features_index.npy")):

        files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            return jsonify({"error": "Database is empty. Please add images to data/dataset/"}), 404
        extractor.index_database(files, FEATURES_DIR)

    database_features = extractor.load_features(FEATURES_DIR)
    if not database_features:
        return jsonify({"error": f"Failed to load index for {descriptor_name}. It might be empty or corrupted due to lack of disk space."}), 500
    
    # 3. Get Top K Neighbors
    voisins = getkVoisins(database_features, query_feature, k=20, distanceName=distance_name)


    
    # 4. Format results & Generate PR Curve
    retrieved_image_paths = [v[0] for v in voisins]
    retrieved_classes = [extract_class_id(p) for f, p, d in [ (v[0], v[0], v[2]) for v in voisins ]] # Helper to get path
    # Actually retrieved_image_paths is already what we need
    retrieved_classes = [extract_class_id(p) for p in retrieved_image_paths]
    query_class = extract_class_id(img_path)
    
    # Count relevant items in DATA_DIR for this class
    all_images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_relevant = len([f for f in all_images if extract_class_id(f) == query_class])
    
    pr_curve = create_plot(len(voisins), retrieved_classes, [query_class], total_relevant)
    
    results = []
    for path, feature, dist in voisins:
        rel_path = os.path.basename(path).replace('.npy', '.jpg')
        results.append({

            "path": f"/images/{rel_path}",
            "score": 1.0 / (1.0 + float(dist))
        })
    
    return jsonify({
        "results": results,
        "pr_curve": pr_curve
    })



@app.route('/search_multimodal', methods=['POST'])
def search_multimodal():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if multimodal_engine.index is None:
        return jsonify({"error": "Multimodal index not loaded. Please run indexing first."}), 404

    # Text-to-Image search
    k = 20
    results = multimodal_engine.search_text_to_image(query, k=k)
    
    # Compute R/P Curve for Multimodal
    from retrieval import extract_class_id
    import numpy as np
    
    # Auto-detect class (mode of top 10 results)
    try:
        top_classes = [extract_class_id(res['path']) for res in results[:10]]
        if top_classes:
            dominant_class = max(set(top_classes), key=top_classes.count)
            # Find all relevant items in dataset for this class
            all_images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            relevant_items = len([f for f in all_images if extract_class_id(f) == dominant_class])
            
            # Compute plot
            retrieved_classes = [extract_class_id(res['path']) for res in results]
            pr_curve_base64 = create_plot(len(results), retrieved_classes, [dominant_class], relevant_items)
            
            # Format results for front-end
            formatted_results = []
            for res in results:
                rel_path = os.path.basename(res['path'])
                formatted_results.append({
                    "path": f"/images/{rel_path}",
                    "score": res['score']
                })
            
            # Add to response
            response_data = {
                "results": formatted_results,
                "pr_curve": pr_curve_base64
            }
            return jsonify(response_data)
    except Exception as e:
        print(f"Error computing multimodal PR: {e}")

    # Fallback if PR fails
    formatted_results = []
    for res in results:
        rel_path = os.path.basename(res['path'])
        formatted_results.append({
            "path": f"/images/{rel_path}",
            "score": res['score']
        })
    return jsonify(formatted_results)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
