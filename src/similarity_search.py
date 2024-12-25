import faiss
import numpy as np
import pickle
import os

class SimilaritySearchEngine:
    def __init__(self, embeddings_path='data/embeddings.pkl'):
        # Load precomputed embeddings
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.image_paths = data['image_paths']

        # Create FAISS index
        dimension = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings))

    def search_similar_images(self, query_embedding, top_k=5):
        # Perform similarity search
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [self.image_paths[idx] for idx in indices[0]], distances[0]