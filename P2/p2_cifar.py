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

# Tensorflow
from tensorflow.compat.v1 import logging

# Modelos y capas
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import BatchNormalization

# Optimizador
from keras.optimizers import SGD

# Función de pérdida
from keras.losses import categorical_crossentropy

# Callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, History

# Conjunto de datos y preprocesamiento
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator

#
# PARÁMETROS GLOBALES
#

N = 25                      # Número de clases
EPOCHS = 100                # Épocas de entrenamiento
BATCH_SIZE = 64             # Tamaño de cada batch de imágenes
SPLIT = 0.2                 # Partición para validación
INPUT_SHAPE = (32, 32, 3)   # Formato de entrada de imágenes
PATIENCE = 5                # Épocas que esperar mientras el modelo no mejora
TAM = (10, 5)               # Tamaño del plot
TEMP = True                 # Generar archivos temporales o definitivos
ACC_NAME = "acc"            # Nombre de la métrica de precisión
DIR = "net/"                # Directorio de trabajo

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

def show_evolution(hist, name):
    """Pinta dos gráficas: una con la evolución de la función de pérdida
       en el conjunto de entrenamiento y en el de validación, y otra con la evolución
       del accuracy en el conjunto de entrenamiento y en el de validación.
        - hist: historial de entrenamiento del modelo.
        - name: nombre del modelo."""

    # Evolución de las funciones de pérdida
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.figure(figsize = TAM)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(["Training loss " + name, "Validation loss " + name])
    plt.show()

    wait()

    # Evolución del accuracy
    acc = hist.history[ACC_NAME]
    val_acc = hist.history["val_" + ACC_NAME]
    plt.figure(figsize = TAM)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(["Training accuracy " + name, "Validation accuracy " + name])
    plt.show()

    wait()

def show_evolution_val(*hist, names):
    """Pinta dos gráficas: una con la evolución de la función de pérdida en el conjunto
       de validación para todos los modelos, y otra con la evolución del accuracy
       en el conjunto de validación para todos los modelos.
        - *hist: historiales de entrenamientos de los modelos.
        - names: lista de nombres correspondientes a los modelos (en orden)."""

    # Evolución de las funciones de pérdida
    plt.figure(figsize = TAM)
    for h in hist:
        val_loss = h.history['val_loss']
        plt.plot(val_loss)

    plt.legend(["Validation loss " + names[i] for i in range(len(hist))])
    plt.show()

    wait()

    # Evolución del accuracy
    plt.figure(figsize = TAM)
    for h in hist:
        val_acc = h.history["val_" + ACC_NAME]
        plt.plot(val_acc)

    plt.legend(["Validation accuracy " + names[i] for i in range(len(hist))])
    plt.show()

    wait()

def show_stats(score, hist, name, show = True):
    """Muestra estadísticas de accuracy y loss y gráficas de evolución.
        - score: métricas de evaluación.
        - hist: historial de entrenamiento.
        - name: nombre del modelo.
        - show: controla si se muestran gráficas con estadísticas."""

    print("\n---------- " + name.upper() + " MODEL EVALUATION ----------")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print()

    # Mostramos gráficas
    if show:
        show_evolution(hist, name)

def show_stats_val(*stats, names, show = True):
    """Muestra estadísticas de accuracy y loss y gráficas de evolución
       en validación.
        - *stats: estadísticas de los modelos. Para cada modelo, se
          consideran sus métricas de evaluación (score) y su historial debug
          entrenamiento. Se mandan primero todos los scores en orden, y
          después todos los historiales.
        - names: lista de nombres correspondientes a los modelos (en orden).
        - show: controla si se muestran gráficas con estadísticas."""

    for i, name in enumerate(names):
        print("\n---------- " + name.upper() + " MODEL EVALUATION ----------")
        print("Test loss:", stats[i][0])
        print("Test accuracy:", stats[i][1])
    print()

    # Mostramos gráficas
    if show:
        show_evolution_val(*stats[len(names):], names = names)

#
# ACCURACY EN EL CONJUNTO DE TEST
#

def accuracy(labels, preds):
    """Deuelve la medida de precisión o 'accuracy' de un modelo sobre el conjunto
       de entrenamiento: porcentaje de etiquetas predichas correctamente frente
       al total.
        - labels: etiquetas correctas en formato matriz binaria.
        - preds: etiquetas predichas en formato matriz binaria."""

    # Convertimos matrices a vectores
    labels = np.argmax(labels, axis = 1)
    preds = np.argmax(preds, axis = 1)

    return sum(labels == preds) / len(labels)

#
# COMPILACIÓN DEL MODELO
#

def compile(model):
    """Definición del optimizador y compilación del modelo."""

    # Definimos el optimizador
    opt = SGD(lr = 0.01, decay = 1e-6,
              momentum = 0.9, nesterov = True)

    # Compilamos el modelo
    model.compile(loss = categorical_crossentropy,
                  optimizer = opt,
                  metrics = [ACC_NAME])

#
# ENTRENAMIENTO DEL MODELO
#

def train(model, model_name, datagen, x_train, y_train,
          save_hist = False, save_weights = False):
    """Entrenar el modelo con los datos de entrenamiento. Devuelve
       el historial de entrenamiento.
        - model, model_name: modelo a entrenar y su nombre.
        - datagen: generador de imágenes de entrenamiento y validación.
        - x_train, y_train: datos de entrenamiento.
        - save_hist: controla si se guarda en fichero el historial de entrenamiento.
        - save_weights: controla si se guardan los mejores pesos obtenidos en fichero."""

    # Nombres de ficheros para guardar estadísticas
    if TEMP:
        file_w = DIR + "temp_" + model_name + "_weights.h5"
        file_h = DIR + "temp_" + model_name + "_hist"
    else:
        file_w = DIR + model_name + "_weights.h5"
        file_h = DIR + model_name + "_hist"

    # Callbacks para el entrenamiento
    callbacks_list = []

    # Paramos si no mejoramos en un número determinado de épocas
    early_stopping_loss = EarlyStopping(monitor = 'val_loss',
                                        patience = PATIENCE,
                                        restore_best_weights = True)
    early_stopping_val = EarlyStopping(monitor = "val_" + ACC_NAME,
                                       patience = PATIENCE,
                                       restore_best_weights = True)

    callbacks_list.append(early_stopping_loss)
    callbacks_list.append(early_stopping_val)

    if save_weights:
        # Vamos guardando los mejores pesos del modelo
        checkpointer = ModelCheckpoint(monitor = "val_" + ACC_NAME,
                                       filepath = file_w,
                                       verbose = 1,
                                       save_weights_only = True,
                                       save_best_only = True)
        callbacks_list.append(checkpointer)

    # Entrenamos el modelo
    hist = model.fit_generator(datagen.flow(x_train,
                                            y_train,
                                            batch_size = BATCH_SIZE,
                                            subset = 'training'),
                               epochs = EPOCHS,
                               steps_per_epoch = len(x_train) * (1 - SPLIT) / BATCH_SIZE,
                               verbose = 1,
                               validation_data = datagen.flow(x_train,
                                                              y_train,
                                                              batch_size = BATCH_SIZE,
                                                              subset = 'validation'),
                               validation_steps = len(x_train) * SPLIT / BATCH_SIZE,
                               callbacks = callbacks_list)

    # Información del entrenamiento
    best_epoch = len(hist.epoch) - PATIENCE
    print("\nNo se ha mejorado en " + str(PATIENCE) + " épocas.")
    print("Mejores pesos obtenidos en la época", best_epoch)

    # Guardamos historial de entrenamiento
    if save_hist:
        with open(file_h, 'wb') as f:
            pickle.dump(hist, f)

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
# EJECUCIÓN COMPLETA DEL MODELO
#

def execute(model_gen, preproc,
            save_s = False, save_h = False, load_w = False, save_w = False):
    """Construir, compilar, entrenar y evaluar un modelo. Devuelve el
       historial de entrenamiento y la evaluación del modelo.
        - model_gen: función que devuelve el modelo en cuestión y su nombre.
        - preproc: decide si se realiza preprocesamiento de datos.
        - save_s: controla si se guardan las estadísticas de evaluación en un fichero.
        - save_h: controla si se guarda el historial de entrenamiento en un fichero.
        - load_w: controla si se cargan los pesos iniciales de un fichero.
        - save_w: controla si se guardan los pesos aprendidos en un fichero."""

    # Construimos y compilamos el modelo
    model, model_name = model_gen()
    compile(model)

    # Nombres de los ficheros para guardar/cargar
    file_w = DIR + model_name + "_weights.h5"
    if TEMP:
        file_s = DIR + "temp_" + model_name + "_score"
    else:
        file_s = DIR + model_name + "_score"

    # Mostramos el modelo
    print(model.summary())

    # Cargamos los datos
    x_train, y_train, x_test, y_test = load_data()

    if preproc:
        # Generador para las imágenes con preprocesamiento y data augmentation
        datagen = ImageDataGenerator(featurewise_center = True,
                                     featurewise_std_normalization = True,
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     zoom_range = 0.2,
                                     horizontal_flip = True,
                                     validation_split = SPLIT)

        # Estandarizamos datos de entrenamiento y test
        datagen.fit(x_train)
        datagen.standardize(x_test)

    else:
        # Generador para las imágenes sin preprocesamiento
        datagen = ImageDataGenerator(validation_split = SPLIT)

    # Cargamos pesos precalculados o entrenamos el modelo
    if load_w:
        model.load_weights(file_w)
        hist = History()
    else:
        hist = train(model, model_name, datagen, x_train, y_train, save_h, save_w)

    # Evaluamos el modelo
    score = evaluate(model, x_test, y_test)

    # Guardamos el resultado de la evaluación
    if save_s:
        with open(file_s, 'wb') as f:
            pickle.dump(score, f)

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

    return model, "basenet"

def ex1(show = True):
    """Ejercicio 1 de la práctica 2. Entrenamiento y evaluación sobre
       CIFAR100 con la red BaseNet.
        - show: controla si se muestran gráficas con estadísticas."""

    # Ejecutamos el modelo
    score, hist = execute(basenet_model, preproc = False,
                          save_s = True, save_h = True, save_w = True)

    # Mostramos estadísticas
    show_stats(score, hist, "basenet", show = show)

#
# APARTADO 2: BASENET MEJORADO
#

def improved_basenet_model():
    """Devuelve el modelo BaseNet mejorado."""

    model = Sequential()

    model.add(Conv2D(32,
                     kernel_size = (3, 3),
                     use_bias = False,
                     input_shape = INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32,
                     kernel_size = (3, 3),
                     use_bias = False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,
                     kernel_size = (3, 3),
                     use_bias = False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64,
                     kernel_size = (3, 3),
                     use_bias = False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,
                    use_bias = False,
                    activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Dense(256,
                    use_bias = False,
                    activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(N,
                    activation = 'softmax'))

    return model, "improved_basenet"

def ex2(show = True):
    """Ejercicio 2 de la práctica 2. Entrenamiento y evaluación sobre
       CIFAR100 con la red BaseNet mejorada.
        - show: controla si se muestran gráficas con estadísticas."""

    # Ejecutamos el modelo
    score, hist = execute(improved_basenet_model, preproc = True,
                          save_s = True, save_h = True, save_w = True)

    # Mostramos estadísticas
    show_stats(score, hist, "improved_basenet", show = show)

def compare(show = True):
    """Comparar el modelo BaseNet con el modelo BaseNet mejorado.
        - show: controla si se muestran gráficas con estadísticas."""

    # Cargamos el historial de entrenamiento de BaseNet
    with open(DIR + "basenet_hist", 'rb') as f1:
        h1 = pickle.load(f1)

    # Cargamos las estadísticas de evaluación de BaseNet
    with open(DIR + "basenet_score", 'rb') as f1:
        s1 = pickle.load(f1)

    # Cargamos el historial de entrenamiento de BaseNet mejorado
    with open(DIR + "improved_basenet_hist", 'rb') as f2:
        h2 = pickle.load(f2)

    # Cargamos las estadísticas de evaluación de BaseNet mejorado
    with open(DIR + "improved_basenet_score", 'rb') as f2:
        s2 = pickle.load(f2)

    # Mostramos estadísticas
    show_stats_val(s1, s2, h1, h2,
                   names = ["basenet", "improved_basenet"], show = show)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la práctica 2 paso a paso."""

    # Do not show Tensorflow warnings
    logging.set_verbosity(logging.ERROR)

    print("\n--- EJERCICIO 1: BASENET ---\n")
    #ex1()

    print("\n--- EJERCICIO 2: BASENET MEJORADO ---\n")
    #ex2()

    print("\n--- COMPARACIÓN: BASENET VS BASENET MEJORADO ---\n")
    compare()

if __name__ == "__main__":
 main()
