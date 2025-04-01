import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

## corre el modelo: carga el modelo, a traves d ela camara con cada gesto que hacemos detecta los keypoints, y con ellos predice
# a que gesto corresponde (clasifica el gesto), y muestra por pantalla el gesto con su clasificacion final.

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Cargar el modelo preentrenado
model = tf.keras.models.load_model('rps_model.h5')

gestos = dict([(0, "Piedra"), (1, "Papel"), (2, "Tijera")])

# Directorio para guardar las imágenes  
output_dir = "imagenes_gestos"  
os.makedirs(output_dir, exist_ok=True)  # Crea el directorio si no existe  

# Contador para guardar imágenes con un nombre único  
contador= 0  

# Función para procesar cada fotograma del video
def process_frame(image):

    global contador  # Para acceder al contador global

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe para encontrar los landmarks
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer los landmarks y crear un array numpy
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

            # Hacer la predicción con el modelo
            prediction = model.predict(np.array([landmarks]))
            predicted_class = np.argmax(prediction)
            print(prediction.round(3), predicted_class) # Para ver las probabilidades de las 3 clases

            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(image, f"Gesto: {gestos[predicted_class]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Guardar la imagen con los landmarks y la predicción  
            filename = os.path.join(output_dir, f"gesto_{contador}.png")  
            cv2.imwrite(filename, image)  
            contador += 1  # Incrementar el contador  

    return image

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()

    # Procesar el fotograma
    processed_frame = process_frame(frame)

    # Mostrar el resultado
    cv2.imshow('Detección de gestos', processed_frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
