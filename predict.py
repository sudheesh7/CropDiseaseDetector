import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ✅ Load your trained model
model = tf.keras.models.load_model(r'C:\project\CropDiseaseDetector\model\crop_disease_model.h5')

# ✅ Correct class names - remove 'PlantVillage' and keep actual labels
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128), color_mode='rgb')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_index


# ✅ Optional: Clean label names (for readable output)
def clean_label(label):
    label = label.replace('_', ' ').replace('___', ' - ')
    return label.title()

# ✅ Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    print(f"📌 Disease Detected: {clean_label(predicted_class)}")
    print(f"📊 Confidence: {confidence:.2f}%")

# ✅ Example usage
predict_image(r'C:\project\CropDiseaseDetector\pepper.JPG')
