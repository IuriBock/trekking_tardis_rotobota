import cv2
from ultralytics import YOLO
import time
import serial

# Inicializa a serial (ajuste 'COM3' para a porta correta)
ser = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)  # Aguarda a porta serial estabilizar

# Carrega o modelo YOLO
model = YOLO("yolov8n.pt")

# Abre a câmera
cap = cv2.VideoCapture(0)

last_sent_time = 0
last_direction = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    frame_center_x = frame.shape[1] // 2

    direction_text = "N"
    
    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r
        class_name = model.names[int(cls)]

        if class_name == "bottle":
            bottle_center_x = int((x1 + x2) // 2)
            dx = bottle_center_x - frame_center_x

            if dx < -30:
                direction_text = 'E'
            elif dx > 30:
                direction_text = 'D'
            else:
                direction_text = 'C'

            current_time = time.time()
            if current_time - last_sent_time > 0.5 and direction_text != last_direction:
                ser.write(direction_text.encode())
                print("Enviado para serial:", direction_text)
                last_sent_time = current_time
                last_direction = direction_text

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (bottle_center_x, int((y1 + y2) // 2)), 5, (0, 0, 255), -1)
            break

    cv2.putText(frame, direction_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    ser.write(direction_text.encode())
    cv2.imshow("Detecção de Garrafa", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
ser.close()
cv2.destroyAllWindows()
