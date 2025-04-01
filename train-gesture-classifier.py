import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
#toma los dtaset , crea un modelo denso, lo entrena con los keypoints(etiqueta), y guarda el modelo en formato h5

# Cargar los datos
def load_data(data_file, label_file):
    data = np.load(data_file, allow_pickle=True)
    labels = np.load(label_file, allow_pickle=True)
    data_coord = data.reshape(-1, 21*2) # 21 landmarks * 2 coordenadas (x, y)
    labels = to_categorical(labels, num_classes=3) # Convertir las etiquetas a one-hot encoding
    return data_coord, labels

# Cargar los datos de entrenamiento y validación
X_train, y_train = load_data('rps_dataset.npy', 'rps_labels.npy')

print(X_train.shape)
print(y_train.shape)
print(X_train[0])
print(y_train[0])
exit()

# Crear el modelo
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=21*2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) # 3 neuronas de salida para 3 clases

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split= 0.1)

# Guardar el modelo en un archivo .h5
model.save('rps_model.h5')
print("Modelo guardado en 'rps_model.h5'.")


# Visualizar resultados de entrenamiento: accuracy y loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 3])
plt.legend(loc='lower left')
plt.show()


