import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('model/breed_model.h5')

# Class names (IMPORTANT - match folder names)
class_names = ['hf', 'jersey', 'red_dane']

# Load image
img_path = 'test.jpg'   # change this to your test image
img = image.load_img(img_path, target_size=(224, 224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)
score = np.max(predictions)
predicted_class = class_names[np.argmax(predictions)]

print(f"Prediction: {predicted_class}")
print(f"Confidence: {score * 100:.2f}%")