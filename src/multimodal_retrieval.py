import torch
import clip
import faiss
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

class MultimodalEngine:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.index = None
        self.image_paths = []

    def encode_images(self, image_list):
        """Encode a list of images using CLIP"""
        features = []
        self.image_paths = image_list
        print(f"Encoding {len(image_list)} images...")
        for img_path in tqdm(image_list):
            image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu().numpy())
        
        features = np.vstack(features).astype('float32')
        return features

    def build_index(self, features):
        """Build FAISS index for fast retrieval"""
        dimension = features.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity with normalized vectors
        self.index.add(features)
        print("FAISS index built.")

    def save_index(self, index_path, paths_path):
        faiss.write_index(self.index, index_path)
        np.save(paths_path, self.image_paths)

    def load_index(self, index_path, paths_path):
        self.index = faiss.read_index(index_path)
        self.image_paths = np.load(paths_path).tolist()

    def search_text_to_image(self, query_text, k=20):
        """Search images using a text query"""
        text = clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        query_vector = text_features.cpu().numpy().astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(k):
            results.append({
                "path": self.image_paths[indices[0][i]],
                "score": float(distances[0][i])
            })
        return results

    def search_image_to_image(self, query_image_path, k=20):
        """Search images using an image query"""
        image = self.preprocess(Image.open(query_image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        query_vector = image_features.cpu().numpy().astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(k):
            results.append({
                "path": self.image_paths[indices[0][i]],
                "score": float(distances[0][i])
            })
        return results
