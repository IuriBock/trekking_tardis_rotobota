import cv2
from ultralytics import YOLO
import time
import numpy as np
#Fique atendo a instalação das bibliotecas a cima. Utilize o comando pip install para realizar suas respectivas instalações.

# Carrega o modelo YOLO
model = YOLO("yolov8n.pt")

# Abre a câmera
cap = cv2.VideoCapture(0)

# Parâmetros da câmera (ajuste conforme sua câmera)
KNOWN_BOTTLE_HEIGHT = 0.22  # Altura média de uma garrafa em metros (22cm)
FOCAL_LENGTH = 1000         # Valor aproximado, precisa ser calibrado

# Função para calcular distância
def calculate_distance(measured_height, frame_height, known_height, focal_length):
    # Calcula a altura em metros na imagem
    image_height = (measured_height / frame_height) * known_height
    # Calcula a distância usando semelhança de triângulos
    distance = (known_height * focal_length) / (measured_height * frame_height / 1000)
    return distance

last_sent_time = 0
last_direction = ""
last_distance = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    frame_center_x = frame.shape[1] // 2
    frame_height = frame.shape[0]

    direction_text = "Objeto não encontrado"
    distance_text = ""
    distance_meters = 0

    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r
        class_name = model.names[int(cls)]

        if class_name == "bottle":
            # Cálculo da posição horizontal
            bottle_center_x = int((x1 + x2) // 2)
            dx = bottle_center_x - frame_center_x

            if dx < -30:
                direction_text = "ESQUERDA"
            elif dx > 30:
                direction_text = "DIREITA"
            else:
                direction_text = "CENTRO"

            # Cálculo da distância
            bottle_height = y2 - y1
            distance_meters = calculate_distance(bottle_height, frame_height, 
                                              KNOWN_BOTTLE_HEIGHT, FOCAL_LENGTH)
            
            # Limita a distância a valores razoáveis (0.3m a 5m)
            distance_meters = np.clip(distance_meters, 0.3, 5)
            distance_text = f"{distance_meters:.2f}m"

            # Limita frequência de impressão
            current_time = time.time()
            if (current_time - last_sent_time > 0.5 and 
                (direction_text != last_direction or 
                 abs(distance_meters - last_distance) > 0.1)):
                print(f"Direção: {direction_text} | Distância: {distance_text}")
                last_sent_time = current_time
                last_direction = direction_text
                last_distance = distance_meters

            # Desenha a caixa e informações
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (bottle_center_x, int((y1 + y2) // 2)), 5, (0, 0, 255), -1)
            
            # Mostra informações na imagem
            info_text = f"{direction_text} | {distance_text}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Mostra altura detectada (para debug)
            cv2.putText(frame, f"Altura: {int(bottle_height)}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            break

    if direction_text == "Objeto não encontrado":
        cv2.putText(frame, direction_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Detecção de Garrafa", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
