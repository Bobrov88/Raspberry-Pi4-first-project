import cv2

index = 0
found_camera = False
while True:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        found_camera = True
        print(f"Камера найдена на индексе {index}")
        cap.release()
    else:
        print(f"Камера не найдена на индексе {index}")
    
    index += 1
    if index > 5:  # Ограничьте количество проверяемых индексов
        break

if not found_camera:
    print("Камеры не найдены.")