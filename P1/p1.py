# coding: utf-8

# Visión por Computador.
# Práctica 1: Filtrado y detección de regiones.
# Antonio Coín Castro.

# Librerías
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Parámetros
IM_PATH = "../img/"  # TODO CAMBIAR POR "imagenes/"
IM1 = IM_PATH + "lena.jpg"
IM2 = IM_PATH + "messi.jpg"

# Bonus 1: Aplicar convolución 2D a partir de dos máscaras 1D (vx, vy)
# Exigimos que las máscaras tengan tamaño impar
def convolution2D(im, vx, vy):
    k = (len(vx) - 1) / 2
    nrows = im.shape[0]
    ncols = im.shape[1]
    im_temp = im.copy()

    for i in range(nrows):
        row = im[i]
        for j in range(int(k)):
            row = np.insert(row, (0, len(row)), 0)
        im_temp[i] = np.convolve(row, vx, 'valid')

    im2 = im_temp.copy()
    for i in range(ncols):
        col = im2[:,i]
        for j in range(int(k)):
            col = np.insert(col, (0, len(col)), 0)
        im_temp[:,i] = np.convolve(col, vy, 'valid')

    return im_temp


# Leemos las imágenes de prueba
im = cv2.imread(IM1, 0)
vx = [1.0/7 for i in range(7)]
vy = [1.0/7 for i in range(7)]
im2 = convolution2D(im, vx, vy)

# Ejercicio 1
im_blur = cv2.GaussianBlur(im, (9,9), 0)
plt.imshow(im_blur, cmap='gray')
plt.waitforbuttonpress()
plt.imshow(im2, cmap='gray')
plt.waitforbuttonpress()
