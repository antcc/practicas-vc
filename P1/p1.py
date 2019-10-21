#! /usr/bin/env python
# coding: utf-8
# uso: ./p1.py

##########################################################################
# Visión por Computador. Curso 2019/20.
# Práctica 1: Filtrado y detección de regiones.
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

IM_PATH = "../img/"            # Carpeta de imágenes # TODO CAMBIAR POR "imagenes/"
IM1 = IM_PATH + "cat.bmp"      # Imagen de ejemplo
EPSILON = 1e-12                # Tolerancia para descomposición SVD
BORDER_CONSTANT = 0            # Tratamiento de bordes #1 en la convolución
BORDER_REPLICATE = 1           # Tratamiento de bordes #2 en la convolución
THRESHOLD = 0.01               # Umbral para máximos en el espacio de escalas
WIDTH, HEIGHT = 7, 7           # Tamaño por defecto del plot
NCOLS_PLOT = 3                 # Número de columnas por defecto en el multiplot

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

def normalize(im):
    """Normaliza una imagen de números reales a [0,1]"""

    return cv2.normalize(im, None, 0.0, 1.0, cv2.NORM_MINMAX)

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

    return im.astype(np.double)

def print_im(im, title = "", show = True, tam = (WIDTH, HEIGHT)):
    """Muestra una imagen cualquiera normalizada.
        - im: imagen a mostrar.
        - show: indica si queremos mostrar la imagen inmediatamente.
        - tam = (width, height): tamaño del plot."""

    show_title = len(title) > 0

    if show:
        fig = plt.figure(figsize = tam)

    im = normalize(im)
    plt.imshow(im, interpolation = None, cmap = 'gray')
    plt.xticks([]), plt.yticks([])

    if show:
        if show_title:
            plt.title(title)
        plt.show(block = False)
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

#
# BONUS 1: implementar convolución 2D a partir de dos máscaras 1D (cuando sea posuble).
#

def convolution2D(im, kernel, border_type = BORDER_REPLICATE, value = 0.0):
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

    return separable_convolution2D(im, vx, vy, border_type, value)

def separable_convolution2D(im, vx, vy, border_type = BORDER_REPLICATE, value = 0.0):
    """Aplica convolución 2D a partir de dos máscaras 1D, y devuelve la imagen resultante.
        - im: imagen sobre la que convolucionar. No se modifica.
        - vx: máscara para las filas. Debe ser de tamaño impar.
        - vy: máscara para las columnas. Debe ser de tamaño impar."""

    # Comprobamos que las máscaras tengan longitud impar
    if len(vx) % 2 == 0 or len(vy) % 2 == 0:
        print("Error: las máscaras 1D deben ser de longitud impar.")
        return np.zeros(im.shape)

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

    nrows = im.shape[0]
    ncols = im.shape[1]

    # Píxeles "extra" en los bordes de cada dimensión
    kx = len(vx) // 2
    ky = len(vy) // 2
    im_res = make_border(im, ky, kx, border_type, value)

    # Aplicamos la máscara por filas
    for i in range(nrows + 2 * ky):
        im_res[i] = convolve(im_res[i], vx, kx)

    # Aplicamos la máscara por columnas
    for j in range(kx, ncols + kx):
        im_res[:,j] = convolve(im_res[:, j], vy, ky)

    return im_res[ky:-ky, kx:-kx]

def make_border(im, vert, horiz, border_type, value):
    """Devuelve una imagen con borde extendido en la dirección horizontal
       y vertical, según la estrategia especificada en 'border_type'.
        - im: imagen original. No se modifica.
        - vert: número de filas extra al inicio y al final de la imagen.
        - horiz: número de columnas extra al inicio y al final de la imagen."""

    nrows = im.shape[0]
    ncols = im.shape[1]

    if border_type == BORDER_CONSTANT:
        im_res = im.copy()

        pad_row = np.full((1, ncols), value, dtype = np.double)
        for i in range(vert):
            im_res = np.vstack([pad_row, im_res, pad_row])

        pad_col = np.full((1, im_res.shape[0]), value, dtype = np.double)
        for j in range(horiz):
            im_res = np.transpose(np.vstack([pad_col, np.transpose(im_res), pad_col]))

    else:
        im_res = np.zeros((nrows + 2 * vert, ncols + 2 * horiz), dtype = np.double)

        for i in range(nrows):
            pad_row_1 = np.full(horiz, im[i][0])
            pad_row_2 = np.full(horiz, im[i][ncols - 1])
            im_res[i + vert] = np.hstack([pad_row_1, im[i], pad_row_2])

        for j in range(ncols):
            pad_col_1 = np.full(vert, im[:, j][0])
            pad_col_2 = np.full(vert, im[:, j][nrows - 1])
            im_res[:, j + horiz] = np.hstack([pad_col_1, im[:, j], pad_col_2])

    return im_res

def convolve(a, v, kx):
    """Devuelve la convolución de dos vectores 1D 'a' y 'v', de longitud len(a), con 'kx'
       posiciones de borde a ambos extremos. Debe ser len(a) >= len(v)."""

    pad = np.zeros(kx, dtype = np.double)
    b = np.hstack([pad, a, pad])

    rows = []
    for i in range(len(a)):
        rows.append(b[i:i + len(v)])
    A = np.array(rows, dtype = np.double)

    return A @ np.transpose(v[::-1])

#
# EJERCICIO 1: filtros Gaussiana, de derivadas y Laplaciana.
#

def gaussian(x, sigma):
    """Evaluación en el punto 'x' de la función Gaussiana de media 0 y desviación típica
       'sigma', sin la constante de normalización."""

    return exp((- x * x) / (2.0 * sigma * sigma))

def gaussian_kernel1D(sigma):
    """Devuelve una máscara Gaussiana 1D normalizada, con desviación típica 'sigma'."""

    # El intervalo para el muestreo (de enteros) será [-3 * sigma, 3 * sigma]
    l = floor(3 * sigma)

    # Calculamos el kernel 1D y normalizamos
    gauss_ker = [gaussian(x, sigma) for x in range(-l, l + 1)]
    return gauss_ker / np.sum(gauss_ker)

def gaussian_blur2D(im, sigma, border_type = BORDER_REPLICATE, value = 0.0):
    """Devuelve el resultado de convolucionar una máscara Gaussiana 2D con una imagen,
       implementada mediante convolución con dos máscaras 1D.
        - im: imagen original. No se modifica.
        - sigma: desviación típica en la dirección horizontal y vertical. Debe ser sigma >= 1/3.
       El tamaño de cada máscara 1D se calcula a partir de la desviación típica:
       2 * ⌊3 * sigma⌋ + 1."""

    if sigma < 1.0 / 3.0:
        return im

    gauss_ker = gaussian_kernel1D(sigma)
    return separable_convolution2D(im, gauss_ker, gauss_ker, border_type, value)

def get_derivatives2D(dx, dy, size):
    """Calcula máscaras de derivadas.
        - dx: orden de la derivada en X.
        - dy: orden de la derivada en Y.
        - size: tamaño de las máscaras. Debe ser impar."""

    return map(lambda x: x.astype(np.double).flatten(), cv2.getDerivKernels(dx, dy, size, normalize = True))

def derivatives2D(im, dx, dy, size, border_type = BORDER_REPLICATE, value = 0.0):
    """Convolucionar una imagen con máscaras de derivadas.
        - dx: orden de la derivada con respecto a X.
        - dy: orden de la derivada con respecto a Y.
        - size: tamaño de las máscaras. Debe ser impar."""

    vx, vy = get_derivatives2D(dx, dy, size)
    return abs(separable_convolution2D(im, vx, vy, border_type, value))

def laplacian2D(im, sigma, size, border_type = BORDER_REPLICATE, value = 0.0):
    """Aplica un filtro Laplaciana-de-Gaussiana a una imagen.
        - im: imagen sobre la que aplicar el filtro. No se modifica.
        - sigma: desviación típica para el alisado Gaussiano.
        - size: tamaño del kernel Laplaciano. Debe ser impar."""

    im_smooth = gaussian_blur2D(im, sigma, border_type, value)
    vxx, v = get_derivatives2D(2, 0, size)
    u, vyy = get_derivatives2D(0, 2, size)

    im1 = separable_convolution2D(im_smooth, vxx, v, border_type, value)
    im2 = separable_convolution2D(im_smooth, u, vyy, border_type, value)
    laplacian = im1 + im2

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
    size = 7
    border = [BORDER_CONSTANT, BORDER_REPLICATE]
    laplacian_im = [im]
    titles = ["Original",
              "Laplaciana-de-Gaussiana 7x7, σ = " + str(sigma[0]) + ", borde constante",
              "Laplaciana-de-Gaussiana 7x7, σ = " + str(sigma[0]) + ", borde replicado",
              "Laplaciana-de-Gaussiana 7x7, σ = " + str(sigma[1]) + ", borde constante",
              "Laplaciana-de-Gaussiana 7x7, σ = " + str(sigma[1]) + ", borde replicado"]

    for s in sigma:
        for b in border:
            laplacian_im.append(laplacian2D(im, s, size, b))

    print_multiple_im(laplacian_im, titles)

#
# EJERCICIO 2: pirámides Gaussiana, Laplaciana y búsqueda de regiones
#

def blur_and_downsample(im, sigma, border_type = BORDER_REPLICATE, value = 0.0):
    """Aplica alisamiento Gaussiano con desviación típica 'sigma' a la imagen 'im',
       y después reduce su tamaño a la mitad."""

    nrows, ncols = im.shape[:2]
    im_blur = gaussian_blur2D(im, sigma, border_type, value)
    im_downsampled = np.array([im_blur[i] for i in range(1, nrows, 2)])
    im_downsampled = np.transpose([im_downsampled[:, j] for j in range(1, ncols, 2)])

    return im_downsampled

def gaussian_pyramid(im, sigma, size, border_type = BORDER_REPLICATE, value = 0.0):
    """Devuelve una lista de imágenes que representan una pirámide Gaussiana.
        - im: imagen original. No se modifica.
        - sigma: desviación típica para el alisado Gaussiano.
        - size: tamaño de la pirámide."""

    pyramid = [im]
    for k in range(size):
        im_next = blur_and_downsample(pyramid[k], sigma, border_type, value)
        pyramid.append(im_next)

        if im_next.shape[0] == 1 or im_next.shape[1] == 1:
            break

    return pyramid

def laplacian_pyramid(im, sigma, size, scale, border_type = BORDER_REPLICATE, value = 0.0):
    """Devuelve una lista de imágenes que representan una pirámide Laplaciana.
        - im: imagen original. No se modifica.
        - sigma: desviación típica para el alisado Gaussiano.
        - size: tamaño de la pirámide.
        - scale: factor para la interpolación bilineal."""

    pyramid = []
    x = im
    for k in range(size):
        # Blur y downsample
        im_blur = gaussian_blur2D(x, sigma, border_type, value)
        im_downsampled = cv2.resize(im_blur, (int(im_blur.shape[1] / scale), int(im_blur.shape[0] / scale)), interpolation = cv2.INTER_NEAREST)

        # Upsample y blur
        im_upsampled = cv2.resize(im_downsampled, (x.shape[1], x.shape[0]), interpolation = cv2.INTER_LINEAR)
        im_upsampled = gaussian_blur2D(im_upsampled, sigma, border_type, value)

        # Añadimos a la pirámide
        pyramid.append(x - im_upsampled)
        x = im_downsampled

        if x.shape[0] == 1 or x.shape[1] == 1:
            break

    # Nos quedamos con frecuencias bajas en el último nivel
    pyramid.append(x)

    return pyramid

def format_pyramid(vim, k):
    """Construye una única imagen en forma de pirámide a partir de varias imágenes,
       cada una con tamaño 1 / 'k' veces el de la anterior."""

    nrows, ncols = vim[0].shape[:2]

    diff = np.sum([im.shape[0] for im in vim[1:]]) - nrows
    extra_rows = diff if diff > 0 else 0
    extra_cols = int(ncols / k)

    if is_grayscale(vim[0]):
        pyramid = np.zeros((nrows + extra_rows, ncols + extra_cols), dtype = np.double)
    else:
        pyramid = np.zeros((nrows + extra_rows, ncols + extra_cols, 3), dtype = np.double)

    pyramid[:nrows, :ncols] = vim[0]

    i_row = 0
    for p in vim[1:]:
        p_nrows, p_ncols = p.shape[:2]
        pyramid[i_row:i_row + p_nrows, ncols:ncols + p_ncols] = p
        i_row += p_nrows

    return pyramid

def reconstruct_im(pyramid, sigma, border_type = BORDER_REPLICATE, value = 0.0):
    """Reconstruye una imagen a partir de su pirámide Laplaciana.
        - pyramid: lista de imágenes que conforman la pirámide.
        - sigma: desviación típica para el alisamiento Gaussiano. Es necesario que sea
          la misma que la que se usó al construir la pirámide."""

    # Extraemos frecuencias bajas
    im = pyramid[-1]

    for p in pyramid[-2::-1]:
        # Upsample y blur
        im = cv2.resize(im, (p.shape[1], p.shape[0]), interpolation = cv2.INTER_LINEAR)
        im = p + gaussian_blur2D(im, sigma, border_type, value)

    return im

def ex2A():
    """Ejemplo de ejecución del ejercicio 2, apartado A."""

    im = read_im(IM1, 1)

    gauss_pyramid1 = format_pyramid(gaussian_pyramid(im, 3, 4, border_type = BORDER_CONSTANT), 2)
    gauss_pyramid2 = format_pyramid(gaussian_pyramid(im, 5, 4), 2)

    print_im(gauss_pyramid1, "Pirámide Gaussiana, σ = 3, borde constante")
    print_im(gauss_pyramid2, "Pirámide Gaussiana, σ = 5, borde replicado")

def ex2B():
    """Ejemplo de ejecución del ejercicio 2, apartado B."""

    im = read_im(IM1, 0)
    sigma = 2
    scale = 2

    laplacian_pyramid1 = laplacian_pyramid(im, sigma, 4, scale)
    rec_im = reconstruct_im(laplacian_pyramid1, sigma)

    print_im(format_pyramid(laplacian_pyramid1, scale),
             title = "Pirámide Laplaciana, σ = " + str(sigma) +
                     ", factor de escala = " + str(scale) + ", borde replicado")

    print_multiple_im([im, rec_im, abs(im - rec_im)],
                      ["Original", "Reconstruida a partir de la pirámide", "Diferencia"])

def blob_detection(im, n, sigma, size, step, border_type = BORDER_REPLICATE, value = 0.0):
    """Realiza detección de regiones en un espacio de escalas a través de
       la Laplaciana-de-Gaussiana, basándose en la técnica de supresión de no-máximos.
        - im: imagen original. Debe estar en escala de grises.
        - n: número de escalas.
        - sigma: desviación típica para el alisado Gaussiano.
        - size: tamaño del kernel Laplaciano.
        - step: factor de incremento de la desviación típica en cada escala."""

    nrows, ncols = im.shape[:2]
    im_color = cv2.normalize(im.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    im_color = cv2.cvtColor(im_color, cv2.COLOR_GRAY2RGB)

    scale_regions = []
    s = sigma
    for p in range(n):
        im_scale = normalize(np.square(laplacian2D(im, s, size, border_type, value)))
        index_lst = []

        # Perform non-maximum supression on the current scale
        for i in range(nrows):
            for j in range(ncols):

                # Select neighbours (counting oneself)
                neighbours = []
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if i + k >= 0 and i + k < nrows and j + l >= 0 and j + l < ncols:
                            neighbours.append(im_scale[i + k][j + l])

                # Non-maximum supression
                if np.max(neighbours) <= im_scale[i, j] and im_scale[i, j] > THRESHOLD:
                    index_lst.append((j, i))

        # Draw circles on current scale
        im_aux = im_color.copy()
        for index in index_lst:
            im_aux = cv2.circle(im_aux, index, int(sqrt(2) * s), (255, 0, 0))

        scale_regions.append(im_aux.astype(np.double))
        s = s * step

    return scale_regions

def ex2C():
    """Ejemplo de ejecución del ejercicio 2, apartado C."""

    im = read_im(IM1, 0)
    n = 6
    sigma = 1.0
    size = 5
    k = 1.2
    titles = ["Escala " + str(i) for i in range(1, n + 1)]

    scale_regions = blob_detection(im, n, sigma, size, k)

    print_multiple_im(scale_regions, titles)

#
# EJERCICIO 3: imágenes híbridas
#

def hybrid_im(im1, im2, sigma1, sigma2, border_type = BORDER_REPLICATE, value = 0.0):
    """Construye una imagen híbrida a partir de las bajas y altas frecuencias
       de dos imágenes del mismo tamaño.
        - im1: imagen de la que se extraen las bajas frecuencias.
        - im2: imagen de la que se extraen las altas frecuencias.
        - sigma1: desviación típica para el filtro paso baja.
        - sigma2: desviación típica para el filtro paso alta.
       Devuelve: bajas frecuencias im1, altas frecuencias im2, imagen híbrida."""

    low_filter = gaussian_kernel1D(sigma1)

    im1_low = separable_convolution2D(im1, low_filter, low_filter, border_type, value)
    im2_high = im2 - separable_convolution2D(im2,low_filter, low_filter, border_type, value)

    return [im1_low, abs(im2_high), abs(im1_low + im2_high)]

def ex3(file1, file2, sigma1, sigma2, color_flag = 0):
    """Ejemplo de ejecución del ejercicio 3.
        - file1: archivo de la primera imagen (frecuencias bajas).
        - file2: archivo de la segunda imagen (frecuencias altas).
        - sigma1: desviación típica para filtro paso baja.
        - sigma2: desviación típica para filtro paso alta.
        - color_flag: indica si las imágenes se leen en color (1) o en grises (0)."""

    # Leemos las imágenes
    im1 = read_im(IM_PATH + file1 + ".bmp", color_flag)
    im2 = read_im(IM_PATH + file2 + ".bmp", color_flag)

    # Datos sobre frecuencias de corte
    CUTOFF = sqrt(2 * log(2))
    cutoff1 = CUTOFF * sigma1
    cutoff2 = CUTOFF * sigma2
    print("\nFrecuencia de corte del filtro paso baja: " + str(cutoff1))
    print("Frecuencia de corte del filtro paso alta: " + str(cutoff2))

    # Creamos la imagen híbrida
    low_freq, high_freq, hybrid = hybrid_im(im1, im2, sigma1, sigma2)
    pyramid = gaussian_pyramid(hybrid, 2, 4)

    # Imprimimos los resultados
    print_multiple_im([low_freq, high_freq, hybrid],
                      ["Bajas frecuencias " + file1, "Altas frecuencias " + file2,
                       "Imagen híbrida " + file1 + " - " + file2])
    print_im(format_pyramid(pyramid, 2),
             "Pirámide Gaussiana de imagen híbrida " + file1 + " - " + file2)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la práctica 1 paso a paso. Cada apartado es una llamada a una función."""

    # Ejercicio 1
    """print("--- EJERCICIO 1 ---")
    print("-- Apartado A")
    ex1A()

    print("-- Apartado B")
    ex1B()

    # Ejercicio 2
    print("\n--- EJERCICIO 2 ---")
    print("-- Apartado A")
    ex2A()

    print("-- Apartado B")
    ex2B()"""

    print("-- Apartado C")
    ex2C()

    # Ejercicio 3
    """print("\n--- EJERCICIO 3 ---")
    print("-- Primera pareja")
    im_low, im_high = "dog", "cat"
    sigma1, sigma2 = 5.0, 7.0
    ex3(im_low, im_high, sigma1, sigma2)

    print("-- Segunda pareja")
    im_low, im_high = "marilyn", "einstein"
    sigma1, sigma2 = 3.0, 7.0
    ex3(im_low, im_high, sigma1, sigma2)

    print("-- Tercera pareja")
    im_low, im_high = "submarine", "fish"
    sigma1, sigma2 = 3.0, 7.0
    ex3(im_low, im_high, sigma1, sigma2)"""

if __name__ == "__main__":
  main()
