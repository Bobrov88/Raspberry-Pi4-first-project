import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключить предупреждения oneDNN

import tensorflow as tf
from tensorflow.keras import layers, models
import json

# Путь к данным (папка Training содержит подпапки с фруктами)
dataset_path = 'fruits-360_100x100/fruits-360/Training'

# Загрузка данных с аугментацией
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(100, 100),  # Изображения 100x100
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=42,
    label_mode='categorical'  # Для совместимости с MobileNetV2
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(100, 100),
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=42,
    label_mode='categorical'
)

# Сохранение названий классов
num_classes = len(train_data.class_names)
with open('class_names.json', 'w') as f:
    json.dump(train_data.class_names, f)

# Аугментация данных (повышает точность)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Использование предобученной MobileNetV2 (оптимизирована для мобильных устройств)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(100, 100, 3),
    include_top=False,  # Не включать верхние слои (заменим своими)
    weights='imagenet'  # Веса, обученные на ImageNet
)
base_model.trainable = False  # Заморозить слои (не обучать заново)

# Сборка модели
model = models.Sequential([
    layers.Input(shape=(100, 100, 3)),  # Входной слой
    data_augmentation,
    layers.Rescaling(1./255),  # Нормализация [0, 255] → [0, 1]
    base_model,
    layers.GlobalAveragePooling2D(),  # Уменьшение размерности
    layers.Dense(256, activation='relu'),  # Полносвязный слой
    layers.Dropout(0.5),  # Регуляризация (борьба с переобучением)
    layers.Dense(num_classes, activation='softmax')  # Выходной слой (вероятности классов)
])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение
model.fit(
    train_data,
    epochs=20,  # Увеличено количество эпох
    validation_data=val_data,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3)  # Остановка при переобучении
    ]
)

# Сохранение модели в формате H5 (для проверки)
model.save('fruit_model.h5')

# Конвертация в TensorFlow Lite (для Raspberry Pi)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('fruit_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Модель обучена и сохранена!")