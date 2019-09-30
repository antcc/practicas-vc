#!/usr/bin/env python
#coding: utf-8

##########################################################################
# Visión por Computador. Curso 2019/20.
# Práctica 1: Filtrado y detección de regiones.
# Antonio Coín Castro.
##########################################################################

#
# LIBRERÍAS
#

import numpy as np
import cv2
from math import ceil
from matplotlib import pyplot as plt

#
# PARÁMETROS GLOBALES
#

IM_PATH = "../img/"            # Carpeta de imágenes # TODO CAMBIAR POR "imagenes/"
IM1 = IM_PATH + "lena.jpg"     # Imagen de ejemplo
WIDTH, HEIGHT = 4, 4           # Tamaño del plot

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse una tecla."""

    input("(Pulsa cualquier tecla para continuar...)")

def is_grayscale(im):
    """Indica si una imagen está en escala de grises."""

    return len(im.shape) == 2

def read_im(filename, color_flag):
    """Devuelve una imagen adecuadamente leída en grises o en color.
        - filename: ruta de la imagen.
        - color_flag: indica si es en color (1) o en grises (0)."""

    im = cv2.imread(filename, color_flag)

    if not is_grayscale(im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # Compatibilidad OpenCV --> Matplotlib

    return im

def print_im(im, show = True):
    """Muestra una imagen cualquiera.
        - im: imagen a mostrar.
        - show: indica si queremos mostrar la imagen inmediatamente. Por defecto es True."""

    if is_grayscale(im):
        plt.imshow(im, cmap = 'gray')
    else:
        plt.imshow(im)

    plt.xticks([]), plt.yticks([]) # Eliminar ejes en el plot
    if show:
        plt.show(block = False)

def print_multiple_im(vim, titles = ""):
    """Muestra una sucesión de imágenes en la misma ventana, eventualmente con sus títulos.
        - vim: sucesión de imágenes a mostrar.
        - titles: o bien vacío o bien una sucesión de títulos del mismo tamaño que vim."""

    show_title = len(titles) > 0

    nrows = ceil(len(vim) / 2.0)
    ncols = 2
    plt.figure(figsize = (WIDTH, HEIGHT))

    for i in range(len(vim)):
        plt.subplot(nrows, ncols, i + 1)
        if show_title:
            plt.title(titles[i])
        print_im(vim[i], False)

    plt.show(block = False)

#
# BONUS 1
#

def convolution2D(im, vx, vy):
    """Aplica convolución 2D a partir de dos máscaras 1D, y devuelve la imagen resultante.
        - im: imagen sobre la que convolucionar. No se modifica.
        - vx: máscara 1D para las filas. Debe ser de tamaño impar.
        - vy: máscara 1D para las columnas. Debe ser de tamaño impar."""

    # Tamaño de la imagen
    nrows = im.shape[0]
    ncols = im.shape[1]

    # Píxeles "extra" en los bordes de cada dimensión
    kx = int((len(vx) - 1) / 2)
    ky = int((len(vy) - 1) / 2)
    zerosx = np.zeros(kx)
    zerosy = np.zeros(ky)

    im_res = np.copy(im) # No modificamos la original

    # Aplicamos la máscara por filas # TODO cambiar por un map #TODO los bordes
    for i in range(nrows):
        row = np.hstack((zerosx, im[i], zerosx))
        im_res[i] = np.convolve(row, vx, 'valid')

    # Aplicamos la máscara por columnas # TODO cambiar por un map #TODO los bordes
    for j in range(ncols):
        col = np.hstack((zerosy, im_res[:,j], zerosy))
        im_res[:,j] = np.convolve(col, vy, 'valid')

    return im_res

#
# EJERCICIO 1
#

def gaussian2D(im, ksize, sX = 0, sY = 0, border = cv2.BORDER_DEFAULT):
    """Devuelve el resultado de aplicar una máscara gaussiana 2D a una imagen.
        - im: imagen original. No se modifica.
        - ksize = (width, height): tupla que indica el tamaño del kernel. Ambos valores
        deben serimpares.
        - sX: desviación típica en la dirección horizontal. Si es 0 se calcula a partir de
        ksize.
        - sY: desviación típica en la dirección vertical. Si es 0 coincide con sX.
        - border: especifica la estrategia a seguir al aplicar la máscara en los bordes.
        Sus posibles valores son: cv.BORDER_CONSTANT, cv.BORDER_REPLICATE, cv.BORDER_REFLECT,
        cv.BORDER_WRAP, cv.BORDER_REFLECT_101, cv.BORDER_TRANSPARENT, cv.BORDER_REFLECT101,
        cv.BORDER_DEFAULT, cv.BORDER_ISOLATED."""

    return cv2.GaussianBlur(im, ksize, sX, sY, border)

def ex1A():
    """Ejemplo de ejecución del ejercicio 1, apartado A."""

    # Leemos las imágenes
    im = read_im(IM1, 1)
    im_gray = read_im(IM1, 0)

    # Gaussiana
    im1 = gaussian2D(im_gray, (9, 9))

    # Derivadas
    vx = [1.0 / 9 for i in range(9)]
    vy = [1.0 / 9 for i in range(9)]
    im2 = convolution2D(im_gray, vx, vy)

    # Imprimimos los resultados
    print_multiple_im([im_gray, im1, im2],
                      ["Original", "Gaussiana 2D 9x9",
                       "Alisamiento 2D uniforme 9x9"])

def ex1B():
    pass

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la práctica 1 paso a paso."""

    print("--- EJERCICIO 1 ---")
    print("Apartado A")
    ex1A()
    wait()
    print("Apartado B")
    ex1B()

if __name__ == "__main__":
  main()
