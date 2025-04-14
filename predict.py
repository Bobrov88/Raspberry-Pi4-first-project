import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключить предупреждения oneDNN

import cv2
import numpy as np
import time
import json
#from tflite_runtime.interpreter import Interpreter  # Используем TFLite
import tensorflow as tf

# Пути к файлам
# MODEL_PATH = 'fruit_model.tflite'
MODEL_PATH = 'fruit_model.h5'
LABELS_PATH = 'class_names.json'
CAMERA_DEVICE = '/dev/video1'  # USB-камера
THRESHOLD = 0.2  # Порог уверенности (настройте под ваши данные)

# Загрузка меток классов
with open(LABELS_PATH, 'r') as f:
    class_names = json.load(f)


model = tf.keras.models.load_model(MODEL_PATH)
# Инициализация модели TFLite
#interpreter = Interpreter(model_path=MODEL_PATH)
#interpreter.allocate_tensors()
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()

#Настройка камеры
cap = cv2.VideoCapture(CAMERA_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Разрешение потока (можно уменьшить для скорости)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра!")
            break
        # Предобработка кадра
        img = cv2.resize(frame, (100, 100))  # Масштабирование до 100x100

############ Read from static pic
#test_image = cv2.imread('test_fruit.jpg')
#img = cv2.resize(test_image, (100,100))
############

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
        img = img.astype(np.float32) / 255.0  # Нормализация [0, 1]
        img = np.expand_dims(img, axis=0)  # Добавление размерности батча (1, 100, 100, 3)
# Подача данных в модель
        #interpreter.set_tensor(input_details[0]['index'], img)
        #interpreter.invoke()
        #prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction = model.predict(img, verbose = 0)
# Постобработка
        #probabilities = np.exp(prediction[0]) / np.sum(np.exp(prediction[0])) # Вероятности классов
        confidence = np.max(prediction[0])  # Максимальная уверенность
        class_id = np.argmax(prediction[0])
# Вывод результата
        if confidence > THRESHOLD:
            label = class_names[class_id]
            elapsed_time = time.time() - start_time
            print(f"Распознано: {label} | Уверенность: {confidence:.2f} | Время: {elapsed_time:.2f} сек")
#         #else:
#          #   print("Фрукт не распознан")

except KeyboardInterrupt:
        print("Остановка по запросу пользователя")



finally:
    cap.release()
    #cv2.destroyAllWindows()