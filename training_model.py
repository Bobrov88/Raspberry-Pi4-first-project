import tensorflow as tf
from tensorflow.keras import layers
import json

path = 'fruits-360_100x100/fruits-360/'
# Загрузка данных
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    image_size=(100, 100),
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=42
)

with open('class_names.json', 'w') as f:
    json.dump(train_data.class_names, f)

# Создание модели
model = tf.keras.Sequential([
    layers.Rescaling(1./255),  # Нормализация пикселей (0-255 → 0-1)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(120)  # 120 классов фруктов/овощей
])

# Обучение
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(train_data, epochs=10)
model.save('fruit_model.h5')