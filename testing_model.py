import cv2
import numpy as np
import tensorflow as tf
import json
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

with open('class_names.json', 'r') as f:
    class_names = json.load(f)
# Загрузка модели
model = tf.keras.models.load_model('fruit_model.h5')

# Загрузка тестового изображения
img = cv2.imread('test_image.jpg')  # положите фото в ту же папку
img = cv2.resize(img, (100, 100))
img = np.expand_dims(img, axis=0)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'fruits-360_100x100/fruits-360/Test/',
    image_size=(100, 100),
    batch_size=64,  # Увеличили batch_size для стабильности
    validation_split=0.2,
    subset='validation',
    seed=42,
    label_mode='categorical' 
)    # Для многоклассовой классификации
# Предсказание
prediction = model.predict(img)
class_id = np.argmax(prediction)
print(f"Это {class_names[class_id]}")