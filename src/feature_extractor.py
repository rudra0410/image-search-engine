import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Load pretrained EfficientNetB0 model without top layers
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
        self.model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

    def extract_features(self, img_path):
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)

        # Extract features
        features = self.model.predict(preprocessed_img)
        return features.flatten()