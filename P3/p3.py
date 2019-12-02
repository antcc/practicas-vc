#! /usr/bin/env python
# coding: utf-8
# uso: ./p1.py

##########################################################################
# Visión por Computador. Curso 2019/20.
# Práctica 3: Detección de puntos relevantes y construcción de panoramas
# Antonio Coín Castro.
##########################################################################

#
# LIBRERÍAS
#

from matplotlib import pyplot as plt
from math import floor, exp, sqrt, log
import numpy as np
import cv2

#
# PARÁMETROS GLOBALES
#

PATH = "img/"         # Carpeta de imágenes
THRESHOLD = 10        # Umbral para máximos en el espacio de escalas
NUM_MAX = 750         # Número de máximos en el espacio de escalas
WIDTH, HEIGHT = 7, 7  # Tamaño por defecto del plot
NCOLS_PLOT = 3        # Número de columnas por defecto en el multiplot

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse una tecla. Limpia el plot anterior"""

    input("(Pulsa cualquier tecla para continuar...)")
    plt.close()

def is_grayscale(im):
    """Indica si una imagen está en escala de grises (monobanda)."""

    return len(im.shape) == 2

def normalize(im):
    """Normaliza una imagen de números reales a [0,1]"""

    return cv2.normalize(im, None, 0.0, 1.0, cv2.NORM_MINMAX)

def read_im(filename, color_flag = 0):
    """Devuelve una imagen de números reales adecuadamente leída en grises o en color.
        - filename: ruta de la imagen.
        - color_flag: indica si es en color (1) o en grises (0)."""

    try:
        im = cv2.imread(filename, color_flag)
        if not is_grayscale(im):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except:
        print("Error: no se ha podido cargar la imagen " + filename)
        quit()

    return im.astype(np.double)

def print_im(im, title = "", show = True, tam = (WIDTH, HEIGHT)):
    """Muestra una imagen cualquiera normalizada.
        - im: imagen a mostrar.
        - show: indica si queremos mostrar la imagen inmediatamente.
        - tam = (width, height): tamaño del plot."""

    show_title = len(title) > 0

    if show:
        fig = plt.figure(figsize = tam)

    im = normalize(im)  # Normalizamos a [0,1]
    plt.imshow(im, interpolation = None, cmap = 'gray')
    plt.xticks([]), plt.yticks([])

    if show:
        if show_title:
            plt.title(title)
        plt.show(block = False)  # No cerrar el plot
        wait()

def print_multiple_im(vim, titles = "", ncols = NCOLS_PLOT, tam = (WIDTH, HEIGHT)):
    """Muestra una sucesión de imágenes en la misma ventana, eventualmente con sus títulos.
        - vim: sucesión de imágenes a mostrar.
        - titles: o bien vacío o bien una sucesión de títulos del mismo tamaño que vim.
        - ncols: número de columnas del multiplot.
        - tam = (width, height): tamaño del multiplot."""

    show_title = len(titles) > 0

    nrows = len(vim) // ncols + (0 if len(vim) % ncols == 0 else 1)
    plt.figure(figsize = tam)

    for i in range(len(vim)):
        plt.subplot(nrows, ncols, i + 1)
        if show_title:
            plt.title(titles[i])
        print_im(vim[i], title = "", show = False)

    plt.show(block = False)
    wait()


def separable_convolution2D(im, vx, vy, border_type = cv2.BORDER_REPLICATE):
    """Aplica convolución 2D a partir de dos máscaras 1D, y devuelve la imagen resultante.
        - im: imagen sobre la que convolucionar. No se modifica.
        - vx: máscara para las filas. Debe ser de tamaño impar.
        - vy: máscara para las columnas. Debe ser de tamaño impar.
        - border_type: tipo de borde para "rellenar" la imagen."""

    # Comprobamos que las máscaras tengan longitud impar
    if len(vx) % 2 == 0 or len(vy) % 2 == 0:
        print("Error: las máscaras 1D deben ser de longitud impar.")
        return np.zeros(im.shape)

    # Aplicamos la convolución por canales
    if is_grayscale(im):
        im_res = channel_separable_convolution2D(im, vx, vy, border_type)
    else:
        channels = cv2.split(im)  # Separar en 3 canales
        im_res = cv2.merge(       # Volver a juntar en una imagen tribanda
                 [channel_separable_convolution2D(ch, vx, vy, border_type)
                  for ch in channels])

    return im_res

def channel_separable_convolution2D(im, vx, vy, border_type):
    """Aplica convolución 2D en un canal partir de dos máscaras 1D, y devuelve
       la imagen resultante.
        - im: imagen monobanda para convolucionar. No se modifica."""

    nrows = im.shape[0]
    ncols = im.shape[1]

    # Píxeles "extra" en los bordes de cada dimensión
    kx = len(vx) // 2
    ky = len(vy) // 2
    im_res = cv2.copyMakeBorder(im, ky, ky, kx, kx, border_type)

    # Aplicamos la máscara por filas
    for i in range(nrows + 2 * ky):
        im_res[i] = np.convolve(im_res[i], vx, 'same')

    # Aplicamos la máscara por columnas
    for j in range(kx, ncols + kx):
        im_res[:,j] = convolve(im_res[:, j], vy, 'same')

    return im_res[ky:-ky, kx:-kx]

def gaussian_pyramid(im, size, border_type = cv2.BORDER_REPLICATE):
    """Devuelve una lista de imágenes que representan una pirámide Gaussiana.
        - im: imagen original. No se modifica.
        - size: tamaño de la pirámide."""

    pyramid = [im]
    for k in range(size):
        pyramid.append(cv2.pyrDown(pyramid[-1], borderType = border_type))

    return pyramid

def format_pyramid(vim, k):
    """Construye una única imagen en forma de pirámide a partir de varias imágenes,
       cada una con tamaño 1 / 'k' veces el de la anterior."""

    nrows, ncols = vim[0].shape[:2]
    diff = np.sum([im.shape[0] for im in vim[1:]]) - nrows
    extra_rows = diff if diff > 0 else 0
    extra_cols = int(ncols / k)

    # Creamos la imagen que hará de marco
    if is_grayscale(vim[0]):
        pyramid = np.zeros((nrows + extra_rows, ncols + extra_cols), dtype = np.double)
    else:
        pyramid = np.zeros((nrows + extra_rows, ncols + extra_cols, 3), dtype = np.double)

    # Colocamos la primera imagen
    pyramid[:nrows, :ncols] = vim[0]

    # Añadimos las siguientes imágenes en su lugar correspondiente
    i_row = 0
    for p in vim[1:]:
        p_nrows, p_ncols = p.shape[:2]
        pyramid[i_row:i_row + p_nrows, ncols:ncols + p_ncols] = p
        i_row += p_nrows

    return pyramid

def blob_detection(im, n, sigma, size, step, border_type = BORDER_REPLICATE, value = 0.0):
    """Realiza detección de regiones en un espacio de escalas a través de
       la Laplaciana-de-Gaussiana, basándose en la técnica de supresión de no-máximos.
        - im: imagen original. Debe estar en escala de grises.
        - n: número de escalas.
        - sigma: desviación típica para el alisado Gaussiano.
        - size: tamaño del kernel Laplaciano.
        - step: factor de incremento de la desviación típica en cada escala."""

    nrows, ncols = im.shape[:2]

    # Pasamos la imagen a color para dibujar círculos de colores sobre ella
    im_color = cv2.normalize(im.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    im_color = cv2.cvtColor(im_color, cv2.COLOR_GRAY2RGB)

    # Construimos el espacio de escalas
    scale_sigma = [sigma * step ** i for i in range(n)]
    scale_regions = [normalize(np.square(laplacian2D(im, s, size, border_type, value))) for s in scale_sigma]

    index_lst = []
    for p in range(n):
        index_lst.append([])
        im_scale = scale_regions[p]

        # Supresión de no máximos en la escala actual
        for i in range(nrows):
            for j in range(ncols):

                # Seleccionamos vecinos en un cubo de lado 3 (contando a uno mismo)
                neighbours = []
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if i + k >= 0 and i + k < nrows and j + l >= 0 and j + l < ncols:
                            neighbours.append(im_scale[i + k, j + l])  # Nivel actual
                            if p > 0:  # Nivel inferior
                                neighbours.append(scale_regions[p - 1][i + k, j + l])
                            if p < n - 1:  # Nivel superior
                                neighbours.append(scale_regions[p + 1][i + k, j + l])

                # Seleccionamos los máximos que superen un umbral
                if im_scale[i, j] > THRESHOLD and np.max(neighbours) <= im_scale[i, j]:
                    index_lst[p].append((i, j))

        # Dibujamos círculos en la imagen original
        blob = im_color.copy()
        for p, lst in enumerate(index_lst):
            lst = sorted(lst, key = lambda x: scale_regions[p][x], reverse = True)

            # Pintamos como mucho NUM_MAX círculos (los más altos de cada escala)
            for index in lst[:NUM_MAX]:

                # Elegimos un color para cada escala (módulo 6) y un radio
                if p % 6 == 0:
                    color = (255, 0, 0)
                elif p % 6 == 1:
                    color = (0, 255, 0)
                elif p % 6 == 2:
                    color = (0, 0, 255)
                elif p % 6 == 3:
                    color = (255, 0, 255)
                elif p % 6 == 4:
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 0)

                radius = int(2 * scale_sigma[p])

                # Pintamos un círculo por cada máximo encontrado
                blob = cv2.circle(blob, index[::-1], radius, color)

    return blob.astype(np.double)

#
# EJERCICIO 1: DETECTOR DE HARRIS
#

def ex1():


    im = read_im(PATH + "Yosemite1.jpg")
    harris_detection(im)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la práctica 3 paso a paso. Cada apartado es una llamada a una función."""

    print("--- EJERCICIO 1: DETECTOR DE HARRIS ---\n")
    ex1()


if __name__ == "__main__":
  main()
