import os
import pickle
from .feature_extractor import FeatureExtractor
import time
from tqdm import tqdm

def precompute_embeddings(image_dir='data/images', output_path='data/embeddings.pkl'):
    # Initialize the feature extractor
    extractor = FeatureExtractor()

    embeddings = []
    image_paths = []

    # Get total number of valid images
    valid_images = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(valid_images)
    
    print(f"\nFound {total_images} images to process")
    
    # Estimate time (assuming ~1 second per image for EfficientNetB0)
    estimated_time = total_images * 1  # 1 second per image
    print(f"Estimated time: {estimated_time//60} minutes and {estimated_time%60} seconds\n")

    # Use tqdm for progress bar
    start_time = time.time()
    for idx, filename in enumerate(tqdm(valid_images, desc="Processing images")):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            try:
                # Show current image being processed
                print(f"\rProcessing image {idx+1}/{total_images}: {filename}", end="")
                
                embedding = extractor.extract_features(img_path)
                embeddings.append(embedding)
                image_paths.append(img_path)
                
                # Calculate and show remaining time
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / (idx + 1)
                remaining_images = total_images - (idx + 1)
                estimated_remaining_time = remaining_images * avg_time_per_image
                
                print(f" | Remaining time: {estimated_remaining_time//60:.0f}m {estimated_remaining_time%60:.0f}s")
                
            except Exception as e:
                print(f"\nError processing {filename}: {e}")

    # Save embeddings and paths
    with open(output_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'image_paths': image_paths}, f)

    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total time taken: {total_time//60:.0f} minutes and {total_time%60:.0f} seconds")
    print(f"Successfully processed {len(embeddings)}/{total_images} images")
    print(f"Embeddings saved to {output_path}")
    
    return embeddings, image_paths

if __name__ == "__main__":
    precompute_embeddings()