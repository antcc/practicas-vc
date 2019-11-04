#! /usr/bin/env python
# coding: utf-8
# uso: ./p2_cifar.py

#############################################################################
# Visión por Computador. Curso 2019/20.
# Práctica 2: Redes neuronales convolucionales. Parte 1: conjunto CIFAR100.
# Antonio Coín Castro.
#############################################################################

#
# LIBRERÍAS
#

# Generales
import numpy as np
import matplotlib.pyplot as plt

# Keras
import keras
import keras.utils as np_utils

# Modelos y capas
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Optimizador
from keras.optimizers import SGD

# Función de pérdida
from keras.losses import categorical_crossentropy

# Conjunto de datos
from keras.datasets import cifar100

#
# PARÁMETROS GLOBALES
#

N = 25           # Número de clases
EPOCHS = 25      # Épocas de entrenamiento
BATCH_SIZE = 32  # Tamaño de cada batch de imágenes

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse una tecla."""

    input("(Pulsa cualquier tecla para continuar...)")

#
# LECTURA Y MODIFICACIÓN DEL CONJUNTO DE IMÁGENES
#

def load_data():
    """Carga el conjunto de datos en 4 vectores: las imágenes de entrenamiento,
       las clases de las imágenes de entrenamiento, las imágenes de test y las
       clases de las imágenes de test.

       Restringimos el conjunto para que tenga solo N clases. Cada imagen tiene
       tamaño (32, 32, 3)."""

    # Leemos y normalizamos los datos
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode = 'fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Nos quedamos con las 25 primeras clases
    train_idx = np.isin(y_train, np.arange(N))
    train_idx = np.reshape(train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    test_idx = np.isin(y_test, np.arange(N))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    # Convertimos los vectores de clases en matrices binarias
    y_train = np_utils.to_categorical(y_train, N)
    y_test = np_utils.to_categorical(y_test, 25)

    return x_train, y_train, x_test, y_test

#
# ACCURACY EN EL CONJUNTO DE TEST
#

def accuracy(labels, preds):
    """Deuelve la medida de precisión o 'accuracy' de un modelo sobre el conjunto
       de entrenamiento: porcentaje de etiquetas predichas correctamente frente
       al total.
        - labels: etiquetas correctas en formato matriz binaria.
        - preds: etiquetas predichas en formato matriz binaria."""

    # Convertir matrices a vectores
    labels = np.argmax(labels, axis = 1)
    preds = np.argmax(preds, axis = 1)

    return sum(labels == preds) / len(labels)

#
# GRÁFICAS DE EVOLUCIÓN Y ESTADÍSTICAS
#

def show_evolution(hist):
    """Pinta dos gráficas, una con la evolución de la función de pérdida en el
       conjunto de entrenamiento y en el de validación, y otra con la evolución
       del accuracy en el conjunto de entrenamiento y en el de validación.
        - hist: historial de entrenamiento del modelo."""

    # Evolución de la función de pérdida
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(["Training loss", "Validation loss"])
    plt.show()

    wait()

    # Evolución del accuracy
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(["Training accuracy", "Validation accuracy"])
    plt.show()

    wait()

def show_stats(score, hist):
    """Muestra estadísticas de accuracy y loss y gráficas de evolución."""

    print("\n ------------- MODEL EVALUATION -------------")
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    print()
    show_evolution(hist)

#
# DESCRIPCIÓN DE LOS MODELOS
#

def basenet_model():
    """Devuelve el modelo de referencia BaseNet."""

    model = Sequential()
    model.add(Conv2D(6, kernel_size = (5, 5),
                     activation = 'relu',
                     input_shape = (32, 32, 3)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(16, kernel_size = (5, 5),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(N, activation = 'softmax'))

    return model

#
# COMPILACIÓN DEL MODELO
#

def compile(model):
    """Definición del optimizador SGD y compilación del modelo."""

    opt = SGD(lr = 0.01, decay = 1e-6,
              momentum = 0.9, nesterov = True)

    model.compile(loss = categorical_crossentropy,
                  optimizer = opt,
                  metrics = ['accuracy'])

#
# ENTRENAMIENTO DEL MODELO
#

def train(model, x_train, y_train, x_test, y_test):
    """Entrenar el modelo con los datos de entrenamiento. Se guarda el estado
       por el que se ha quedado entrenando. Devuelve el historial de entrenamiento."""

    hist = model.fit(x_train, y_train,
                     batch_size = BATCH_SIZE,
                     epochs = EPOCHS,
                     verbose = 1,
                     validation_data = (x_test, y_test))

    return hist

#
# PREDICCIÓN Y EVALUACIÓN SOBRE EL CONJUNTO DE TEST
#

def predict(model, x_test):
    """Predicción de etiquetas sobre el conjunto de test."""

    preds = model.predict(x_test)

    return preds

def evaluate(model, x_test, y_test):
    """Evaluar el modelo sobre el conjunto de test."""

    score = model.evaluate(x_test, y_test, verbose = 0)

    return score

#
# APARTADO 1: BASENET
#

def ex1():
    """Ejercicio 1 de la práctica 2. Entrenamiento y evaluación sobre
       CIFAR100 con la red BaseNet."""

    # Construimos y compilamos el modelo
    basenet = basenet_model()
    compile(basenet)

    # Guardamos los pesos iniciales antes de entrenar
    ini_weights = basenet.get_weights()

    # Cargamos los datos y entrenamos el modelo
    x_train, y_train, x_test, y_test = load_data()
    hist = train(basenet, x_train, y_train, x_test, y_test)

    # Evaluamos el modelo
    score = evaluate(basenet, x_test, y_test)

    # Mostramos estadísticas
    show_stats(score, hist)

    # Guardamos pesos entrenados en un fichero en formato HDF5
    file_w = "basenet_weights_" + str(EPOCHS) + ".h5"
    basenet.save_weights(file_w)
    print("\nPesos guardados en el fichero " + file_w)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la práctica 2 paso a paso."""

    print("--- EJERCICIO 1: BASENET ---")
    ex1()

if __name__ == "__main__":
  main()
