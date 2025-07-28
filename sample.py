import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ✅ Load the trained model (Keras format or HDF5)
model = tf.keras.models.load_model(r'C:\project\CropDiseaseDetector\model\crop_disease_model.h5')

# ✅ This must exactly match the order of folders in training
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# ✅ Prediction Function
def predict_image(img_path):
    # Load image with the same target size used in training
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    print(f"\n✅ Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")

# ✅ Try with your image
predict_image(r'C:\project\CropDiseaseDetector\pepper.JPG')
