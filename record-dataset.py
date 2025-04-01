
#e toma de la camara el gesto de una sola mano marcando los landmarks, una vez que muestre los landmarks
# uno tiene que oprimir 0, 1 o 2 para clasificar cada gesto como piedra, papel o tijera. Oprimir q para grabar los 2 archivos del dataset y sale del programa
# pip install mediapipe opencv-python numpy
#grabar los dataset
import cv2  
import numpy as np  
import mediapipe as mp  

# Inicializa MediaPipe Hands  
mp_hands = mp.solutions.hands  
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)  
mp_drawing = mp.solutions.drawing_utils  

# Inicializa las listas para almacenar los datos  
landmark_data = []  
labels = []  

# Función para obtener los landmarks  
def get_hand_landmarks(image):  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    results = hands.process(image_rgb)  
    if results.multi_hand_landmarks:  
        for hand_landmarks in results.multi_hand_landmarks:  
            landmarks = []  
            for landmark in hand_landmarks.landmark:  
                landmarks.append((landmark.x, landmark.y))

            # Dibujar las hand connections en la imagen de la mano
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            return landmarks  
    return None  

# Captura de video  
cap = cv2.VideoCapture(0)  

contador= 0

print("Presiona '0' para piedra, '1' para papel, '2' para tijeras.")  
print("Presiona 'q' para salir y guardar el dataset.")  

while cap.isOpened():  
    ret, frame = cap.read()  
    if not ret:  
        print("Error al capturar el frame.")  
        break  
    
    landmarks = get_hand_landmarks(frame) 
    
    cv2.imshow('Hand Tracking', frame)  
   
    key = cv2.waitKey(1)  
    
    if key == ord('0'):  # Piedra  
        if landmarks is not None:  
            landmark_data.append(landmarks)  
            labels.append(0)  
            contador +=1
            print(f"Piedra guardada., Cantidad gestos guardados: {contador}")  
    elif key == ord('1'):  # Papel  
        if landmarks is not None:  
            landmark_data.append(landmarks)  
            labels.append(1) 
            contador +=1
            print(f"Papel guardado.,  Cantidad gestos guardados: {contador}") 
    elif key == ord('2'):  # Tijeras  
        if landmarks is not None:  
            landmark_data.append(landmarks)  
            labels.append(2)  
            contador +=1
            print(f"Tijera guardada.,  Cantidad gestos guardados: {contador}") 
    elif key == ord('q'):  # Salir  
        break  


# Guarda los datos en archivos .npy  
np.save('rps_dataset.npy', landmark_data)  
np.save('rps_labels.npy', labels)  

# Libera la captura y cierra las ventanas  
cap.release()  
cv2.destroyAllWindows()

