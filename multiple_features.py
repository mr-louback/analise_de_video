import cv2
import numpy as np

# Carregar o modelo MobileNet SSD
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Capturar o vídeo (substitua 'video.mp4' pelo caminho do seu vídeo)
cap = cv2.VideoCapture('videos_0/video_10.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pré-processamento da imagem
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Configuração de entrada
    net.setInput(blob)

    # Realizar detecção de objetos
    detections = net.forward()

    # Loop sobre as detecções
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Se a confiança for maior que um limiar (por exemplo, 0.5)
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Desenhar a caixa delimitadora e rótulo
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Adicionar rótulo específico para objetos, pessoas e carros
            if class_id == 15:  # Código da classe para pessoa
                label = "Pessoa"
            elif class_id == 6 or class_id == 7:  # Códigos de classe para carro e ônibus
                label = "Carro/Ônibus"
            else:
                label = "Objeto Desconhecido"

            # Adicionar o rótulo ao frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar o frame resultante
    cv2.imshow('Detecção de Objetos', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos ao terminar
cap.release()
cv2.destroyAllWindows()
