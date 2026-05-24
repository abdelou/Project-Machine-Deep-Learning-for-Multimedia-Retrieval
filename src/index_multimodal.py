import os
import argparse
from multimodal_retrieval import MultimodalEngine

def index_flickr8k(dataset_path, output_dir):
    """
    Index Flickr8k images using CLIP and FAISS.
    dataset_path: path to the folder containing Images
    """
    engine = MultimodalEngine()
    
    # List all images in the dataset
    allowed_exts = ('.jpg', '.jpeg', '.png')
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(allowed_exts)]
    
    print(f"Found {len(image_paths)} images.")
    
    # Encode and index
    features = engine.encode_images(image_paths)
    engine.build_index(features)
    
    # Save index
    os.makedirs(output_dir, exist_ok=True)
    engine.save_index(
        os.path.join(output_dir, "flickr8k_clip.index"),
        os.path.join(output_dir, "flickr8k_paths.npy")
    )
    print(f"Index saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index Flickr8k dataset")
    parser.add_argument("--path", required=True, help="Path to Flickr8k images folder")
    parser.add_argument("--out", default="outputs/multimodal", help="Output directory")
    args = parser.parse_args()
    
    index_flickr8k(args.path, args.out)
