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
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Keras
import keras
import keras.utils as np_utils

# Modelos y capas
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Flatten, MaxPooling2D, Activation
from keras.layers import BatchNormalization

# Optimizador
from keras.optimizers import SGD

# Función de pérdida
from keras.losses import categorical_crossentropy

# Conjunto de datos y preprocesamiento
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator

#
# PARÁMETROS GLOBALES
#

N = 25                     # Número de clases
EPOCHS = 25                # Épocas de entrenamiento
BATCH_SIZE = 32            # Tamaño de cada batch de imágenes
SPLIT = 0.1                # Partición para validación
INPUT_SHAPE = (32, 32, 3)  # Formato de entrada de imágenes

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
    y_test = np_utils.to_categorical(y_test, N)

    return x_train, y_train, x_test, y_test

#
# GRÁFICAS DE EVOLUCIÓN Y ESTADÍSTICAS
#

def show_evolution(hist):
    """Pinta dos gráficas: una con la evolución de la función de pérdida
       en el conjunto de entrenamiento y en el de validación, y otra con la evolución
       del accuracy en el conjunto de entrenamiento y en el de validación.
        - hist: historial de entrenamiento del modelo."""

    # Evolución de las funciones de pérdida
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

def show_evolution_val(*hist):
    """Pinta dos gráficas: una con la evolución de la función de pérdida en el conjunto
       de validación para todos los modelos, y otra con la evolución del accuracy
       en el conjunto de validación para todos los modelos.
        - *hist: lista de historiales de entrenamientos de los modelos."""

    # Evolución de las funciones de pérdida
    for h in hist:
        val_loss = h.history['val_loss']
        plt.plot(val_loss)

    plt.legend(["Validation loss " + str(i + 1) for i in range(len(hist))])
    plt.show()
    wait()

    # Evolución del accuracy
    for h in hist:
        val_acc = h.history['val_accuracy']
        plt.plot(val_acc)

    plt.legend(["Validation accuracy " + str(i + 1) for i in range(len(hist))])

    plt.show()
    wait()

def show_stats(score, hist):
    """Muestra estadísticas de accuracy y loss y gráficas de evolución."""

    print("\n ------------- MODEL EVALUATION -------------")
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    print()
    show_evolution(hist)

def show_stats_val(score, *hist):
    """Muestra estadísticas de accuracy y loss y gráficas de evolución
       en la validación."""

    print("\n ------------- MODEL EVALUATION -------------")
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    print()
    show_evolution_val(*hist)


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

def train(model, x_train, y_train, x_test, y_test, save_hist = False):
    """Entrenar el modelo con los datos de entrenamiento. Se guarda el estado
       por el que se ha quedado entrenando. Devuelve el historial de entrenamiento.
        - save_hist: controla si se guarda en fichero el historial de entrenamiento."""

    hist = model.fit(x_train, y_train,
                     batch_size = BATCH_SIZE,
                     epochs = EPOCHS,
                     verbose = 1,
                     validation_data = (x_test, y_test))
    if save_hist:
        with open("basenet_hist_" + str(EPOCHS), 'wb') as f:
            pickle.dump(hist, f)

    return hist

def train_gen(model, datagen, x_train, y_train, x_test, y_test, save_hist = False):
    """Entrenar el modelo con los datos de entrenamiento a partir de
       un generador de imágenes. Los parámetros y los valores devueltos son iguales
       que los de la función anterior."""

    hist = model.fit_generator(datagen.flow(x_train,
                                            y_train,
                                            batch_size = BATCH_SIZE),
                               epochs = EPOCHS,
                               steps_per_epoch = len(x_train) / BATCH_SIZE,
                               validation_data = (x_test, y_test),
                               validation_steps = len(x_test) / BATCH_SIZE)

    if save_hist:
        with open("basenet_hist_" + str(EPOCHS), 'wb') as f:
            pickle.dump(hist, f)

    return hist

#
# EVALUACIÓN SOBRE EL CONJUNTO DE TEST
#

def evaluate(model, x_test, y_test):
    """Evaluar el modelo sobre el conjunto de test."""

    score = model.evaluate(x_test, y_test, verbose = 0)

    return score

#
# EJECUCIÓN COMPLETA DEL MODELO
#

def execute(model_gen, filename = "", load_w = False, save_w = False):
    """Construir, compilar, entrenar y evaluar un modelo. Devuelve el
       historial de entrenamiento y la evaluación del modelo.
        - model_gen: función que devuelve el modelo en cuestión.
        - filename: nombre del fichero HDF5 para cargar/guardar pesos.
        - load_w: controla si se cargan los pesos iniciales de un fichero.
        - save_w: controla si se guardan los pesos aprendidos en un fichero."""

    # Construimos y compilamos el modelo
    model = model_gen()
    compile(model)

    # Cargamos pesos precalculados
    if load_w:
        model.load_weights(filename)

    # Cargamos los datos y entrenamos el modelo
    x_train, y_train, x_test, y_test = load_data()
    hist = train(model, x_train, y_train, x_test, y_test)

    # Evaluamos el modelo
    score = evaluate(model, x_test, y_test)

    # Guardamos los pesos aprendidos
    if save_w:
        model.save_weights(filename)

    return score, hist

def execute_gen(model_gen, filename = "", load_w = False, save_w = False):
    """Construir, compilar, entrenar y evaluar un modelo a partir de un
       generador de imágenes. Los parámetros y los valores devueltos son iguales
       que los de la función anterior."""

    # Construimos y compilamos el modelo
    model = model_gen()
    compile(model)

    # Cargamos pesos precalculados
    if load_w:
        model.load_weights(filename)

    # Cargamos los datos y creamos los un generador
    x_train, y_train, x_test, y_test = load_data()
    datagen = ImageDataGenerator(featurewise_center = True,
                                 featurewise_std_normalization = True)

    # Estandarizar datos de entrenamiento y test
    datagen.fit(x_train)
    datagen.standardize(x_test)

    # Entrenamos el modelo
    hist = train_gen(model, datagen, x_train, y_train, x_test, y_test)

    # Evaluamos el modelo
    score = evaluate(model, x_test, y_test)

    # Guardamos los pesos aprendidos
    if save_w:
        model.save_weights(filename)

    return score, hist

#
# APARTADO 1: BASENET
#

def basenet_model():
    """Devuelve el modelo de referencia BaseNet."""

    model = Sequential()

    model.add(Conv2D(6,
                     kernel_size = (5, 5),
                     activation = 'relu',
                     input_shape = INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(16,
                     kernel_size = (5, 5),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(50,
                    activation = 'relu'))
    model.add(Dense(N,
                    activation = 'softmax'))

    return model

def ex1(show = True):
    """Ejercicio 1 de la práctica 2. Entrenamiento y evaluación sobre
       CIFAR100 con la red BaseNet.
        - show: controla si se muestran las estadísticas de evaluación."""

    score, hist = execute(basenet_model)

    # Mostramos estadísticas
    if show:
        show_stats(score, hist)

#
# APARTADO 2: BASENET MEJORADO
#

def improved_basenet_model():
    """Devuelve el modelo BaseNet mejorado."""

    model = Sequential()

    model.add(Conv2D(6,
                     kernel_size = (5, 5),
                     use_bias = False,
                     input_shape = INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(16,
                     kernel_size = (5, 5),
                     use_bias = False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(50,
                    use_bias = False,
                    activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(N,
                    activation = 'softmax'))

    return model

def ex2(show = True):
    """Ejercicio 2 de la práctica 2. Entrenamiento y evaluación sobre
       CIFAR100 con la red BaseNet mejorada.
        - show: controla si se muestran las estadísticas de evaluación."""

    score, hist = execute_gen(improved_basenet_model)

    # Mostramos estadísticas
    if show:
        show_stats(score, hist)

def compare():
    """Comparar el modelo BaseNet con el modelo BaseNet mejorado."""

    # Cargar el historial de entrenamiento de BaseNet
    with open("basenet_hist_25", 'rb') as f:
        h1 = pickle.load(f)

    # Ejecutar BaseNet mejorado
    s2, h2 = execute_gen(basenet_model)

    show_stats_val(s2, h1, h2)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la práctica 2 paso a paso."""

    """print("\n--- EJERCICIO 1: BASENET ---\n")
    ex1()"""

    print("\n--- EJERCICIO 2: BASENET MEJORADO ---\n")
    compare()

if __name__ == "__main__":
  main()
