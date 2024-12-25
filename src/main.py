import streamlit as st
from PIL import Image
from feature_extractor import FeatureExtractor
from similarity_search import SimilaritySearchEngine

def main():
    st.title('Image Similarity Search')

    # Upload query image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the uploaded image
        query_img = Image.open(uploaded_file)

        # Resize and display the query image
        query_img_resized = query_img.resize((263, 385))
        st.image(query_img_resized, caption='Uploaded Image', use_container_width=False)

        # Feature extraction and similarity search
        if st.button("Search Similar Images"):
            with st.spinner("Analyzing query image..."):
                try:
                    # Initialize feature extractor and search engine
                    extractor = FeatureExtractor()
                    search_engine = SimilaritySearchEngine()

                    # Save the uploaded image temporarily
                    query_img_path = 'temp_query_image.jpg'
                    query_img.save(query_img_path)

                    # Extract features from the query image
                    query_embedding = extractor.extract_features(query_img_path)

                    # Perform similarity search
                    similar_images, distances = search_engine.search_similar_images(query_embedding)

                    # Display similar images
                    st.subheader('Similar Images')
                    cols = st.columns(len(similar_images))
                    for i, (img_path, dist) in enumerate(zip(similar_images, distances)):
                        with cols[i]:
                            similar_img = Image.open(img_path).resize((375, 550))
                            st.image(similar_img, caption=f'Distance: {dist:.2f}', use_container_width=True)

                except Exception as e:
                    st.error(f"Error during similarity search: {e}")

if __name__ == '__main__':
    main()
