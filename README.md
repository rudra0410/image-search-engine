## Image Similarity Search Engine
A deep learning-based image similarity search engine that uses EfficientNetB0 for feature extraction and FAISS for fast similarity search. The application provides a web interface built with Streamlit for easy interaction.

Features
- Deep Feature Extraction: Uses EfficientNetB0 (pre-trained on ImageNet) to extract meaningful features from images
- Fast Similarity Search: Implements FAISS for efficient nearest-neighbor search
- Interactive Web Interface: User-friendly interface built with Streamlit
- Real-time Processing: Shows progress and time estimates during feature extraction
- Scalable Architecture: Designed to handle large image datasets efficiently

## Installation
## Prerequisites

Python 3.8 or higher
pip package manager

## Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/image-similarity-search.git
cd image-similarity-search
```
2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. Install required packages:
```
pip install -r requirements.txt
```

## Project Structure
```
image-similarity-search/
├── data/
│   ├── images/                     # Directory for train dataset images
│   ├── sample-test-images/         # Directory for test dataset images
│   └── embeddings.pkl              # Pre-computed image embeddings
├── src/
│   ├── feature_extractor.py    # EfficientNetB0 feature extraction
│   ├── preprocessing.py        # Image preprocessing and embedding computation
│   ├── similarity_search.py    # FAISS-based similarity search
│   └── main.py                 # Streamlit web interface
├── requirements.txt
├── README.md
└── .gitignore
```
## Usage

1. **Prepare Your Dataset:**
Get training image dataset from drive:
```
https://drive.google.com/file/d/1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN/view?usp=drive_link
```
Place your image dataset in the data/images directory
Supported formats: JPG, JPEG, PNG

2. **Generate Embeddings:**
```
python -m src.preprocessing
```

**This will**:
- Process all images in the dataset
- Show progress and time estimates
- Save embeddings to data/embeddings.pkl

3. **Run the Web Interface:**
```
streamlit run src/main.py
```

4. Using the Interface:

- Upload a query image using the file uploader
- Click "Search Similar Images"
- View the most similar images from your dataset



## Technical Details
**Feature Extraction**
- Uses EfficientNetB0 without top layers
- Input image size: 224x224 pixels
- Output feature dimension: 1280

**Similarity Search**
- Uses FAISS IndexFlatL2 for L2 distance-based search
- Returns top-k most similar images (default k=5)

**Web Interface**
- Responsive design with Streamlit
- Displays original and similar images with similarity scores
- Progress tracking during processing

**Dependencies**
- TensorFlow 2.x
- FAISS-cpu (or FAISS-gpu for GPU support)
- Streamlit
- Pillow
- NumPy
- tqdm

**Performance**
- Feature extraction: ~1 second per image on CPU
- Similarity search: Near real-time for datasets up to 100k images
- Memory usage depends on dataset size (approximately 5KB per image embedding)
