#! /usr/bin/env python
# coding: utf-8
# uso: ./p2_resnet.py

#############################################################################
# Visión por Computador. Curso 2019/20.
# Práctica 2: Redes neuronales convolucionales. Parte 2: red ResNet50.
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

# Tensorflow
from tensorflow.compat.v1 import logging

# Modelos y capas
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense
from keras.layers import Flatten, Dropout, BatchNormalization, Activation

# Optimizador
from keras.optimizers import SGD, RMSprop

# Función de pérdida
from keras.losses import categorical_crossentropy

# ResNet
from keras.applications.resnet50 import ResNet50, preprocess_input

# Lectura y preprocesamiento de datos
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

#
# PARÁMETROS GLOBALES
#

N = 200                               # Número de clases
EPOCHS_1 = 50                         # Épocas de entrenamiento para ejercicio 1
EPOCHS_2 = 15                         # Épocas de entrenamiento para ejercicio 2
BATCH_SIZE = 64                       # Tamaño de cada batch de imágenes
SPLIT = 0.1                           # Partición para validación
INPUT_SIZE = (224, 224)               # Dimensiones de las imágenes de entrada
OUTPUT_SIZE = (2048,)                 # Dimensiones de las imágenes de salida
INPUT_SHAPE = INPUT_SIZE + (3,)       # Formato de entrada de imágenes
OUTPUT_SHAPE = (7, 7) + OUTPUT_SIZE   # Formato de salida de imágenes
TAM = (10, 5)                         # Tamaño del plot
ACC_NAME = "acc"                      # Nombre de la métrica de precisión

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse una tecla."""

    input("(Pulsa cualquier tecla para continuar...)")

#
# LECTURA Y MODIFICACIÓN DEL CONJUNTO DE IMÁGENES
#

def read_im(names):
    """Lee las imágenes cuyos nombres están especificados en un vector de entrada.
       Devuelve las imágenes en un vector y sus clases en otro.
        - names: vector con los nombres (ruta relativa) de las imágenes."""

    classes = np.array([im.split('/')[0] for im in names])
    vim = np.array([img_to_array(load_img(DIR + im, target_size = INPUT_SIZE))
                    for im in names])

    return vim, classes

def load_data():
    """Carga el conjunto de datos en 4 vectores: las imágenes de entrenamiento,
       las clases de las imágenes de entrenamiento, las imágenes de test y las
       clases de las imágenes de test.

       Lee las imágenes y las clases de los ficheros 'train.txt' y 'test.txt'."""

    # Cargamos los ficheros
    train_images = np.loadtxt(DIR + "train.txt", dtype = str)
    test_images = np.loadtxt(DIR + "test.txt", dtype = str)

    # Leemos las imágenes
    train, train_classes = read_im(train_images)
    test, test_classes = read_im(test_images)

    # Convertimos las clases a números enteros
    unique_classes = np.unique(np.copy(train_classes))
    for i in range(len(unique_classes)):
      train_classes[train_classes == unique_classes[i]] = i
      test_classes[test_classes == unique_classes[i]] = i

    # Convertimos los vectores de clases en matrices binarias
    train_classes = np_utils.to_categorical(train_classes, N)
    test_classes = np_utils.to_categorical(test_classes, N)

    # Barajamos los datos
    train_perm = np.random.permutation(len(train))
    train = train[train_perm]
    train_classes = train_classes[train_perm]

    test_perm = np.random.permutation(len(test))
    test = test[test_perm]
    test_classes = test_classes[test_perm]

    return train, train_classes, test, test_classes

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

def show_stats(score, hist, name, show = True):
    """Muestra estadísticas de accuracy y loss y gráficas de evolución.
        - score: métricas de evaluación.
        - hist: historial de entrenamiento.
        - name: nombre del modelo.
        - show: controla si se muestran gráficas con estadísticas."""

    print("\n---------- " + name.upper() + " EVALUATION ----------")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print()

    # Mostramos gráficas
    if show:
        show_evolution(hist, name)

#
# CÁLCULO DE PRECISIÓN EN EL CONJUNTO DE TEST
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

def compile(model, lr = 0.01, use_sgd = True):
    """Definición del optimizador y compilación del modelo.
        - model: modelo a compilar.
        - lr: learning rate.
        - use_sgd: decide si se optimiza con 'sgd' (True) o con
          'rmsprop' (False)."""

    # Definimos el optimizador
    if use_sgd:
      opt = SGD(lr = lr, decay = 1e-6,
                momentum = 0.9, nesterov = True)
    else:
      opt = RMSprop(lr = lr)

    # Compilamos el modelo
    model.compile(loss = categorical_crossentropy,
                  optimizer = opt,
                  metrics = [ACC_NAME])

#
# ENTRENAMIENTO DEL MODELO
#

def train(model, epochs, x_train, y_train):
    """Entrenar el modelo con los datos de entrenamiento. Devuelve
       el historial de entrenamiento.
        - model: modelo a entrenar.
        - epochs: épocas de entrenamiento.
        - x_train, y_train: datos de entrenamiento."""

    # Entrenamos el modelo
    hist = model.fit(x_train, y_train,
                     batch_size = BATCH_SIZE,
                     epochs = epochs,
                     verbose = 1,
                     validation_split = SPLIT)

    return hist

def train_gen(model, datagen, epochs, x_train, y_train):
    """Entrenar el modelo con los datos de entrenamiento a partir de
       un generador de imágenes. Devuelve el historial de entrenamiento.
        - model: modelo a entrenar.
        - epochs: épocas de entrenamiento.
        - x_train, y_train: datos de entrenamiento."""

    hist = model.fit_generator(datagen.flow(x_train,
                                            y_train,
                                            batch_size = BATCH_SIZE,
                                            subset = 'training'),
                               epochs = epochs,
                               steps_per_epoch = len(x_train) * (1 - SPLIT) / BATCH_SIZE,
                               verbose = 1,
                               validation_data = datagen.flow(x_train,
                                                              y_train,
                                                              batch_size = BATCH_SIZE,
                                                              subset = 'validation'),
                               validation_steps = len(x_train) * SPLIT / BATCH_SIZE)

    return hist

#
# PREDICIÓN Y EVALUACIÓN
#

def predict_gen(model, datagen, x):
    """Predicción de etiquetas sobre un conjunto de imágenes.
        - model: modelo a usar para predecir.
        - datagen: generador de imágenes.
        - x: conjunto de datos para predecir su clase."""

    preds = model.predict_generator(datagen.flow(x,
                                                 batch_size = 1,
                                                 shuffle = False),
                                    verbose = 1,
                                    steps = len(x))

    return preds

def evaluate(model, x_test, y_test):
    """Evaluar el modelo sobre el conjunto de test.
        - model: modelo a usar para evaluar.
        - x_test, y_test: datos de test."""

    score = model.evaluate(x_test, y_test, verbose = 0)

    return score

def evaluate_gen(model, datagen, x_test, y_test):
    """Evaluar el modelo sobre el conjunto de test, usando un generador
       de imágenes.
        - model: modelo a usar para evaluar.
        - datagen: generador de imágenes.
        - x_test, y_test: datos de test."""

    score = model.evaluate_generator(datagen.flow(x_test,
                                                  y_test,
                                                  batch_size = 1,
                                                  shuffle = False),
                                      verbose = 0,
                                      steps = len(x_test))

    return score

#
# EJECUCIÓN COMPLETA DEL MODELO
#

def execute(model_gen, epochs, x_train, y_train, x_test, y_test):
    """Compilar, entrenar y evaluar un modelo. Devuelve el
       historial de entrenamiento y la evaluación del modelo.
        - model_gen: función que devuelve el modelo en cuestión.
        - epochs: épocas de entrenamiento.
        - x_train, y_train: datos de entrenamiento.
        - x_test, y_test: datos de test."""

    # Construimos el modelo
    model = model_gen()

    # Compilamos el modelo
    compile(model)

    # Mostramos el modelo
    print(model.summary())

    # Entrenamos el modelo
    hist = train(model, epochs, x_train, y_train)

    # Evaluamos el modelo
    score = evaluate(model, x_test, y_test)

    return score, hist

#
# EJERCICIO 1: RED PREENTRENADA COMO EXTRACTOR DE CARACTERÍSTICAS
#

def fc_model():
    """Devuelve un modelo fully-connected básico."""

    model = Sequential()

    model.add(Dense(1024,
                    activation = 'relu',
                    input_shape = OUTPUT_SIZE))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.7))

    model.add(Dense(N, activation = 'softmax'))

    return model

def conv_model():
    """Devuelve un modelo convolucional básico."""

    model = Sequential()

    model.add(Conv2D(64,
                     kernel_size = (3, 3),
                     use_bias = False,
                     activation = 'relu',
                     input_shape = OUTPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024,
                    activation = 'relu',
                    use_bias = False))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    model.add(Dense(N,
                    activation = 'softmax'))

    return model

def resnet_feature_extraction(use_conv = False, show = True):
    """Usar ResNet50 preentrenada en ImageNet como un extractor de características,
       para confeccionar un modelo que clasfique imágenes del conjunto Caltech-UCSD.
        - use_conv: tipo de modelo usado para "rellenar" la red. Puede ser True
          para añadir capas convolucionales o False para añadir directamente capas
          fully-connected.
        - show: controla si se muestran gráficas con estadísticas."""

    # Cargamos los datos
    print("Leyendo imágenes...\n")
    x_train, y_train, x_test, y_test = load_data()

    # Creamos un generador para entrenamiento y otro para test
    datagen_train = ImageDataGenerator(preprocessing_function = preprocess_input)
    datagen_test = ImageDataGenerator(preprocessing_function = preprocess_input)

    # Decidimos el formato de salida de la red ResNet50
    if use_conv:
        pooling_type = None   # Salida: tensor 7 x 8 x 2048
        model_gen = conv_model
    else:
        pooling_type = 'avg'  # Salida: vector de 2048 entradas
        model_gen = fc_model

    # Usamos ResNet50 preentrenada en ImageNet sin la última capa
    resnet = ResNet50(weights = 'imagenet',
                     include_top = False,
                     pooling = pooling_type,
                     input_shape = INPUT_SHAPE)

    # Extraemos características de las imágenes con el modelo anterior
    print("Extrayendo características...\n")
    features_train = predict_gen(resnet, datagen_train, x_train)
    features_test = predict_gen(resnet, datagen_test, x_test)

    # Ejecutamos un modelo con las características extraídas como entrada
    score, hist = execute(model_gen, EPOCHS_1,
                          features_train, y_train, features_test, y_test)

    # Mostramos estadísticas
    if show:
        show_stats(score, hist, model_gen.__name__, show = show)

#
# EJERCICIO 2: FINE-TUNING EN RED PREENTRENADA
#

def fc_model_2(x):
    """Devuelve un modelo fully-connected básico (instancia de Model).
        - x: capa sobre la que construir el modelo."""

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(N, activation='softmax')(x)

    return output

def resnet_fine_tuning(show = True, save_w = False, load_w = False):
    """Usar ResNet50 preentrenada en ImageNet para hacer fine tuning y
       confeccionar un modelo que clasfique imágenes del conjunto Caltech-UCSD.
        - save_w: controla si se guardan los pesos aprendidos en un fichero.
        - load_w: controla si se cargan pesos precalculados desde un fichero.
        - show: controla si se muestran gráficas con estadísticas."""

    # Nombre de fichero para guardar/cargar pesos
    file_w = "/content/drive/My Drive/resnet_weights.h5"

    # Cargamos los datos
    print("Leyendo imágenes...\n")
    x_train, y_train, x_test, y_test = load_data()

    # Creamos un generador para entrenamiento y otro para test
    datagen_train = ImageDataGenerator(preprocessing_function = preprocess_input,
                                       validation_split = SPLIT)
    datagen_test = ImageDataGenerator(preprocessing_function = preprocess_input)

    # Usamos ResNet50 preentrenada en ImageNet sin la última capa
    resnet = ResNet50(weights = 'imagenet',
                      include_top = False,
                      pooling = 'avg',
                      input_shape = INPUT_SHAPE)

    # Definimos un nuevo modelo a partir de ResNet50
    output = fc_model_2(resnet.output)
    model = Model(inputs=resnet.input, outputs=output)

    # Mostramos el modelo
    print(model.summary())

    # Compilamos el modelo
    compile(model)

    # Cargamos pesos precalculados
    if load_w:
      print("\nCargando pesos precalculados de " + file_w)
      model.load_weights(file_w)

    # Entrenamos el modelo
    hist = train_gen(model, datagen_train, EPOCHS_2, x_train, y_train)

    # Guardamos los pesos aprendidos
    if save_w:
      print("\nGuardando pesos aprendidos en " + file_w)
      model.save_weights(file_w)

    # Evaluamos el modelo
    print("\nEvaluando modelo...")
    score = evaluate_gen(model, datagen_test, x_test, y_test)

    # Mostramos estadísticas
    if show:
      show_stats(score, hist, "fine_tuning", show = True)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la segunda parte de la práctica 2 paso a paso."""

    # No mostrar warnings de TensorFlow
    logging.set_verbosity(logging.ERROR)

    print("\n--- EJERCICIO 1: EXTRACTOR DE CARACTERÍSTICAS ---\n")
    resnet_feature_extraction()

    print("\n--- EJERCICIO 2: FINE-TUNING ---\n")
    resnet_fine_tuning()

if __name__ == "__main__":
 main()
