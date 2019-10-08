#! /usr/bin/env python
# coding: utf-8

##########################################################################
# Visión por Computador. Curso 2019/20.
# Práctica 1: Filtrado y detección de regiones.
# Antonio Coín Castro.
##########################################################################

#
# LIBRERÍAS
#

from matplotlib import pyplot as plt
from math import floor, exp
import numpy as np
import cv2

#
# PARÁMETROS GLOBALES
#

IM_PATH = "../img/"            # Carpeta de imágenes # TODO CAMBIAR POR "imagenes/"
IM1 = IM_PATH + "cat.jpg"      # Imagen de ejemplo
EPSILON = 1e-12                # Tolerancia para descomposición SVD
BORDER_CONSTANT = 0            # Tratamiento de bordes #1 en la convolución
BORDER_REPLICATE = 1           # Tratamiento de bordes #2 en la convolución
WIDTH, HEIGHT = 7, 7           # Tamaño del multiplot
NCOLS_PLOT = 3                 # Número de columnas en el multiplot

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse una tecla. Limpia el plot anterior"""

    input("(Pulsa cualquier tecla para continuar...)")
    plt.close()

def is_grayscale(im):
    """Indica si una imagen está en escala de grises."""

    return len(im.shape) == 2

def read_im(filename, color_flag):
    """Devuelve una imagen adecuadamente leída en grises o en color.
        - filename: ruta de la imagen.
        - color_flag: indica si es en color (1) o en grises (0)."""

    try:
        im = cv2.imread(filename, color_flag)
        if not is_grayscale(im):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except:
        print("Error: no se ha podido cargar la imagen " + filename)
        quit()

    return im

def print_im(im, show = True):
    """Muestra una imagen cualquiera normalizada.
        - im: imagen a mostrar.
        - show: indica si queremos mostrar la imagen inmediatamente. Por defecto es True."""

    im = im.astype(float)
    im = cv2.normalize(im, None, 0, 1, cv2.NORM_MINMAX)

    if is_grayscale(im):
        plt.imshow(im, cmap = 'gray')
    else:
        plt.imshow(im)

    plt.xticks([]), plt.yticks([])
    if show:
        plt.show(block = False)

def print_multiple_im(vim, titles = ""):
    """Muestra una sucesión de imágenes en la misma ventana, eventualmente con sus títulos.
        - vim: sucesión de imágenes a mostrar.
        - titles: o bien vacío o bien una sucesión de títulos del mismo tamaño que vim."""

    show_title = len(titles) > 0

    nrows = len(vim) // NCOLS_PLOT + (0 if len(vim) % NCOLS_PLOT == 0 else 1)
    plt.figure(figsize = (WIDTH, HEIGHT))

    for i in range(len(vim)):
        plt.subplot(nrows, NCOLS_PLOT, i + 1)
        if show_title:
            plt.title(titles[i])
        print_im(vim[i], False)

    plt.show(block = False)

#
# BONUS 1: implementar convolución 2D a partir de dos máscaras 1D (cuando sea posuble).
#

def convolution2D(im, kernel, border_type = BORDER_REPLICATE, value = 0):
    """Aplica convolución 2D con un kernel (si es separable), y devuelve la imagen
       resultante.
        - im: imagen sobre la que convolucionar. No se modifica.
        - kernel: matriz que actúa como máscara.
        - border_type: especifica la estrategia a seguir al aplicar una máscara en los
        bordes de la imagen. Puede ser BORDER_CONSTANT ó BORDER_REPLICATE.
        - value: si border_type = BORDER_CONSTANT, indica el color del borde."""

    # Comprobamos si el kernel tiene rango 1 mediante descomposición SVD
    u, s, v = np.linalg.svd(kernel)
    rank = np.sum(s > EPSILON)

    if rank != 1:
        print("Error: el kernel proporcionado no tiene rango 1, y por tanto no es separable.")
        return np.zeros(im.shape)

    # Obtenemos las máscaras separadas
    vx = u[:,0]
    vy = v[:,0]

    if len(vx) % 2 == 0 or len(vy) % 2 == 0:
        print("Error: las máscaras 1D deben ser de longitud impar.")
        return np.zeros(im.shape)

    return separable_convolution2D(im, vx, vy, border_type, value)

def separable_convolution2D(im, vx, vy, border_type = BORDER_REPLICATE, value = 0):
    """Aplica convolución 2D a partir de dos máscaras 1D, y devuelve la imagen resultante.
        - im: imagen sobre la que convolucionar. No se modifica.
        - vx: máscara para las filas. Debe ser de tamaño impar.
        - vy: máscara para las columnas. Debe ser de tamaño impar."""

    # Aplicamos la convolución por canales
    if is_grayscale(im):
        im_res = channel_separable_convolution2D(im, vx, vy, border_type, value)
    else:
        channels = cv2.split(im)
        im_res = cv2.merge(
                 [channel_separable_convolution2D(ch, vx, vy, border_type, value)
                  for ch in channels])

    return im_res

def channel_separable_convolution2D(im, vx, vy, border_type, value):
    """Aplica convolución 2D en un canal partir de dos máscaras 1D, y devuelve
       la imagen resultante.
        - im: imagen monobanda para convolucionar. No se modifica."""

    # Tamaño de la imagen
    nrows = im.shape[0]
    ncols = im.shape[1]

    # Píxeles "extra" en los bordes de cada dimensión
    kx = len(vx) // 2
    ky = len(vy) // 2
    im_res = make_border(im, ky, kx, border_type, value).astype(float)

    # Aplicamos la máscara por filas
    for i in range(ky, nrows + ky):
        im_res[i] = np.convolve(im_res[i], vx, 'same')

    # Aplicamos la máscara por columnas
    for j in range(kx, ncols + kx):
        im_res[:,j] = np.convolve(im_res[:,j], vy, 'same')

    return im_res[ky:-ky, kx:-kx]

def make_border(im, vert, horiz, border_type, value):
    """Devuelve una imagen con borde extendido en la dirección horizontal
       y vertical, según la estrategia especificada.
        - im: imagen original. No se modifica.
        - vert: número de filas extra al inicio y al final de la imagen.
        - horiz: número de columnas extra al inicio y al final de la imagen."""

    nrows = im.shape[0]
    ncols = im.shape[1]

    if border_type == BORDER_CONSTANT:
        im_res = im.copy()

        pad_row = np.full((1, ncols), value)
        for i in range(vert):
            im_res = np.vstack([pad_row, im_res, pad_row])

        pad_col = np.full((1, im_res.shape[0]), value)
        for j in range(horiz):
            im_res = np.transpose(np.vstack([pad_col, np.transpose(im_res), pad_col]))

    else:
        im_res = np.zeros((nrows + 2 * vert, ncols + 2 * horiz))

        for i in range(nrows):
            pad_row_1 = np.full(horiz, im[i][0])
            pad_row_2 = np.full(horiz, im[i][ncols - 1])
            im_res[i + vert] = np.hstack([pad_row_1, im[i], pad_row_2])

        for j in range(ncols):
            pad_col_1 = np.full(vert, im[:,j][0])
            pad_col_2 = np.full(vert, im[:,j][nrows - 1])
            im_res[:,(j + horiz)] = np.hstack([pad_col_1, im[:,j], pad_col_2])

    return im_res

#
# EJERCICIO 1: filtros gaussiana, derivadas y laplaciana.
#

def gaussian(x, sigma):
    """Función gaussiana de media 0 y desviación típica sigma. No tiene el factor
       1 / sqrt(2 * pi) * sigma porque luego se normalizará."""

    return exp((-1 * x * x) / (2 * sigma * sigma))

def gaussian_blur2D(im, sigma, border_type = BORDER_REPLICATE, value = 0):
    """Devuelve el resultado de convolucionar una máscara gaussiana 2D con una imagen,
       implementada mediante convolución con dos máscaras 1D.
        - im: imagen original. No se modifica.
        - sigma: desviación típica en la dirección horizontal y vertical.
       El tamaño de cada máscara 1D se calcula a partir de la desviación típica:
       2 * ⌊3 * sigma⌋ + 1."""

    # El intervalo para el muestreo (de enteros) será [-3 * sigma, 3 * sigma]
    l = 3 * floor(sigma)

    # Calculamos el kernel 1D y normalizamos
    gauss_ker = [gaussian(x, sigma) for x in range(-l, l + 1)]
    gauss_ker = gauss_ker / np.sum(gauss_ker)

    return separable_convolution2D(im, gauss_ker, gauss_ker, border_type, value)

def get_derivatives2D(dx, dy, size):
    """Calcula máscaras de derivadas.
        - dx: orden de la derivada en X.
        - dy: orden de la derivada en Y.
        - size: tamaño de las máscaras. Debe ser impar."""

    return map(lambda x: x.flatten(), cv2.getDerivKernels(dx, dy, size, normalize = True))

def derivatives2D(im, dx, dy, size, border_type = BORDER_REPLICATE, value = 0):
    """Convolucionar una imagen con máscaras de derivadas.
        - dx: orden de la derivada con respecto a X.
        - dy: orden de la derivada con respecto a Y.
        - size: tamaño de las máscaras. Debe ser impar."""

    vx, vy = get_derivatives2D(dx, dy, size)
    return abs(separable_convolution2D(im, vx, vy, border_type, value))

def laplacian2D(im, sigma, size, border_type = BORDER_REPLICATE, value = 0):
    """Aplica un filtro Laplaciana-de-Gaussiana a una imagen.
        - im: imagen sobre la que aplicar el filtro. No se modifica.
        - sigma: desviación típica para el alisado gaussiano.
        - size: tamaño del kernel laplaciano. Debe ser impar."""

    im_smooth = gaussian_blur2D(im, sigma, border_type, value)
    vxx, v = get_derivatives2D(2, 0, size)
    u, vyy = get_derivatives2D(0, 2, size)

    im1 = separable_convolution2D(im_smooth, vxx, v, border_type, value)
    im2 = separable_convolution2D(im_smooth, u, vyy, border_type, value)
    laplacian = sigma * sigma * (im1 + im2)

    return abs(laplacian)

def ex1A():
    """Ejemplo de ejecución del ejercicio 1, apartado A."""

    im = read_im(IM1, 1)
    im_gray = read_im(IM1, 0)

    # Gaussiana
    im1 = gaussian_blur2D(im, 1, BORDER_CONSTANT, 0)
    im2 = gaussian_blur2D(im, 3)
    im3 = gaussian_blur2D(im, 7)

    # Derivadas
    im4 = derivatives2D(im_gray, 1, 0, 3)
    im5 = derivatives2D(im_gray, 0, 1, 3)
    im6 = derivatives2D(im_gray, 0, 2, 3)

    # Imprimimos los resultados
    print("Gaussiana")
    print_multiple_im([im, im1, im2, im3],
                      ["Original",
                       "GaussianBlur σ = 1, borde constante",
                       "GaussianBlur σ = 3, borde replicado",
                       "GaussianBlur σ = 7, borde replicado"])

    wait()

    print("Derivadas")
    print_multiple_im([im_gray, im4, im5, im6],
                      ["Original (grises)",
                       "Derivada resp. X 3x3, borde constante",
                       "Derivada resp. Y 3x3, borde replicado",
                       "Derivada 2ª resp. Y 3x3, borde replicado"])

def ex1B():
    """Ejemplo de ejecución del ejercicio 1, apartado B."""

    im = read_im(IM1, 0)
    sigma = [1, 3]
    size = 3
    border = [BORDER_CONSTANT, BORDER_REPLICATE]
    laplacian_im = [im]
    titles = ["Original",
              "Laplaciana-de-Gaussiana 3x3, σ = " + str(sigma[0]) + ", borde constante",
              "Laplaciana-de-Gaussiana 3x3, σ = " + str(sigma[0]) + ", borde replicado",
              "Laplaciana-de-Gaussiana 3x3, σ = " + str(sigma[1]) + ", borde constante",
              "Laplaciana-de-Gaussiana 3x3, σ = " + str(sigma[1]) + ", borde replicado"]

    for s in sigma:
        for b in border:
            laplacian_im.append(laplacian2D(im, s, size, b))

    print_multiple_im(laplacian_im, titles)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la práctica 1 paso a paso."""

    """print("--- EJERCICIO 1 ---")
    print("-- Apartado A")
    ex1A()
    wait()"""
    print("-- Apartado B")
    ex1B()
    wait()
    print("\n--- EJERCICIO 2 ---")

if __name__ == "__main__":
  main()
