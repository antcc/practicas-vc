#! /usr/bin/env python
# coding: utf-8
# uso: ./p3.py

##########################################################################
# Visión por Computador. Curso 2019/20.
# Práctica 3: Detección de puntos relevantes y construcción de panoramas.
# Antonio Coín Castro.
##########################################################################

#
# LIBRERÍAS
#

from matplotlib import pyplot as plt
from math import log
import numpy as np
import cv2
import random

#
# PARÁMETROS GLOBALES
#

PATH = "imagenes/"    # Carpeta de imágenes
WINDOW_SIZE = 3       # Tamaño de ventana para esquinas
WINDOW_SIZE_F = 7     # Tamaño de ventana para supresión de no máximos
THRESHOLD = 10        # Umbral para máximos en el espacio de escalas
NUM_MAX = 1100        # Número de máximos en el espacio de escalas
ZOOM = 5              # Factor de zoom
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

def gray2rgb(im):
    """Convierte una imagen a RGB."""

    im_rgb = cv2.normalize(im.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(im_rgb, cv2.COLOR_GRAY2RGB)

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

    return im.astype(np.float32)

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

def gaussian_pyramid(im, size, border_type = cv2.BORDER_REPLICATE):
    """Devuelve una lista de imágenes que representan una pirámide Gaussiana.
        - im: imagen original. No se modifica.
        - size: tamaño de la pirámide."""

    pyramid = [im]
    for k in range(size):
        pyramid.append(cv2.pyrDown(pyramid[-1], borderType = border_type))

    return pyramid

def format_pyramid(vim, k = 2):
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

def pad_im(im):
    """Completa una imagen con 0 hasta tener un ancho y alto potencia de 2."""

    nrows, ncols = im.shape[:2]

    rows_pad = log(nrows, 2)
    if int(rows_pad) != rows_pad:
        rows_pad = 2 ** (int(rows_pad) + 1) - nrows
    else:
        rows_pad = 0

    cols_pad = log(ncols, 2)
    if int(cols_pad) != cols_pad:
        cols_pad = 2 ** (int(cols_pad) + 1) - ncols
    else:
        cols_pad = 0

    return cv2.copyMakeBorder(im, 0, rows_pad, 0, cols_pad, cv2.BORDER_CONSTANT)

def non_maximum_supression(f, window_size):
    """Realiza supresión de no máximos en una imagen. Develve una lista con los índices
       de los puntos que son máximos locales.
        - f: imagen sobre la que trabajar.
        - window_size: tamaño del vecindario de cada punto. Debe ser impar."""

    nrows, ncols = f.shape[:2]
    max_index = []
    d = window_size // 2

    for x in range(nrows):
        for y in range(ncols):
            # Si no supera el umbral, lo saltamos
            if f[x, y] <= THRESHOLD:
                continue

            # Seleccionamos vecinos en un cuadrado de lado 'window_size'
            t = x - d if x - d >= 0 else 0
            b = x + d + 1
            l = y - d if y - d >= 0 else 0
            r = y + d + 1
            window = f[t:b, l:r]
            max_window = np.amax(window)

            # Comprobamos si es máximo local
            if max_window <= f[x, y]:
                max_index.append((x, y))
                window[:] = 0
                f[x, y] = max_window

    return max_index

#
# EJERCICIO 1: DETECTOR DE HARRIS
#

def harris_detection(im, levels):
    """Realiza la detección de esquinas según el descriptor de Harris. Devuelve las imágenes
       con los puntos dibujados en cada escala, la imagen original con todos los puntos,
       y un vector de los puntos detectados.
        - im: imagen original en escala de grises.
        - levels: niveles para la pirámide Gaussiana."""

    im_scale_kp = []
    keypoints_orig = []
    window = WINDOW_SIZE_F
    num_points = NUM_MAX

    # Pirámide gaussiana de la imagen completada con 0 hasta tener tamaño potencia de 2
    orig_rows, orig_cols = im.shape[:2]
    im_pad = pad_im(im)
    pyramid = gaussian_pyramid(im_pad, levels)

    # Derivadas de la imagen y sus pirámides gaussianas
    im_blur = cv2.GaussianBlur(im_pad, ksize = (0, 0), sigmaX = 4.5)
    im_dx = gaussian_pyramid(cv2.Sobel(im_blur, -1, 1, 0), levels)
    im_dy = gaussian_pyramid(cv2.Sobel(im_blur, -1, 0, 1), levels)

    # Detectamos puntos en cada escala
    for s in range(len(pyramid)):
        im_scale = pyramid[s]
        nrows, ncols = im_scale.shape[:2]

        # Extraemos información de cada píxel
        dst = cv2.cornerEigenValsAndVecs(im_scale, WINDOW_SIZE, ksize = 3)

        # Calculamos la función "corner strength"
        f = np.empty_like(im_scale)
        for x in range(nrows):
            for y in range(ncols):
                l1 = dst[x, y, 0]
                l2 = dst[x, y, 1]
                f[x, y] = (l1 * l2) / (l1 + l2) if l1 + l2 != 0.0 else 0.0

        # Realizamos supresión de no máximos, ordenados por intensidad
        max_index = non_maximum_supression(f, window)
        max_index = sorted(max_index, key = lambda x: f[x], reverse = True)

        # Nos quedamos con los 'num_points' puntos de mayor intensidad
        keypoints = []
        for p in max_index[:num_points]:
            # Calculamos la orientación de los puntos
            norm = np.sqrt(im_dx[s][p] * im_dx[s][p] + im_dy[s][p] * im_dy[s][p])
            angle_sin = im_dy[s][p] / norm if norm > 0 else 0.0
            angle_cos = im_dx[s][p] / norm if norm > 0 else 0.0
            angle = np.degrees(np.arctan2(angle_sin, angle_cos)) + 180

            # Creamos una estructura KeyPoint con cada punto que sobrevive
            keypoints.append(cv2.KeyPoint(p[1], p[0], WINDOW_SIZE * (levels - s + 1) / 1.3, _angle = angle))

            # También con sus coordenadas respecto a la imagen original
            keypoints_orig.append(cv2.KeyPoint((2 ** s) * p[1], (2 ** s) * p[0], WINDOW_SIZE * (levels - s + 1), _angle = angle))

        # Pasamos la imagen a color y dibujamos cada KeyPoint
        im_rgb = gray2rgb(im_scale)
        im_kp = cv2.drawKeypoints(im_rgb, keypoints, np.array([]),
                                  flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        print("Puntos detectados en la octava " + str(s) + ": " + str(len(keypoints)))

        # Guardamos el resultado de la octava actual
        im_scale_kp.append(im_kp.astype(np.float32)[:orig_rows // 2 ** s, :orig_cols // 2 ** s])

        # Actualizamos el tamaño de ventana y el número de puntos a detectar
        if window > 3:
            window = window - 2
        if num_points > 400:
            num_points = int(num_points / 1.5)

    # Pintamos todos los KeyPoints en la imagen original
    im_rgb = gray2rgb(im)
    im_kp = cv2.drawKeypoints(im_rgb, keypoints_orig, np.array([]),
                              flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_kp = im_kp.astype(np.float32)[:orig_rows, :orig_cols]

    print("Puntos totales detectados: " + str(len(keypoints_orig)) + "\n")

    return im_scale_kp, im_kp, keypoints_orig

def refine_corners(im, keypoints):
    """Calcula las coordenadas subpixel de puntos que representan esquinas. Devuelve una
       lista de 3 imágenes interpoladas (en un entorno 10x10 con zoom de 5x) con las
       coordenadas originales y las refinadas de un punto (cada una).
        - im: imagen original en escala de grises.
        - keypoints: puntos sobre los que corregir las coordenadas."""

    res = []
    win_size = (3, 3)
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    points = np.array([p.pt for p in keypoints], dtype = np.uint32)
    corners = points.reshape(len(keypoints), 1, 2).astype(np.float32)

    # Calculamos las coordenadas subpixel de los KeyPoints
    cv2.cornerSubPix(im, corners, win_size, zero_zone, criteria)

    # Elegimos tres puntos aleatoriamente cuyas coordenadas difieran
    selected_points = []
    count = 0
    while count < 3:
        index = random.randint(0, len(points) - 1)
        if index not in selected_points and \
           (points[index][:2] != corners[index][0][:2]).any():
            selected_points.append(index)
            count = count + 1

    for index in selected_points:
        # Recuperamos las coordenadas originales e interpoladas
        y, x = points[index][:2]
        ry, rx = corners[index][0][:2]

        # Pasamos la imagen original a color para dibujar sobre ella
        im_rgb = gray2rgb(im).astype(np.float32)

        # Seleccionamos una ventana 11x11 alrededor del punto original
        t = x - 5 if x - 5 >= 0 else 0
        b = x + 5 + 1
        l = y - 5 if y - 5 >= 0 else 0
        r = y + 5 + 1
        window = im_rgb[t:b, l:r]

        # Interpolamos con zoom de 5x
        window = cv2.resize(window, None, fx = ZOOM, fy = ZOOM)

        # Dibujamos en rojo el punto original en el centro
        window = cv2.circle(window, (ZOOM * 5 + 1, ZOOM * 5 + 1), 3, (255, 0, 0))

        # Dibujamos en verde el punto corregido
        window = cv2.circle(window, (int(ZOOM * (5 + ry - y) + 1), int(ZOOM * (5 + rx - x) + 1)), 3, (0, 255, 0))

        res.append(window)

    return res

def ex1():
    """Ejemplo de ejecución del ejercicio 1."""

    im1 = read_im(PATH + "yosemite1.jpg")
    im2 = read_im(PATH + "yosemite2.jpg")

    print("Detectando puntos Harris en yosemite1.jpg...\n")
    h1_scale, h1_orig, h1_keypoints = harris_detection(im1, 3)
    print_im(format_pyramid(h1_scale))
    print_im(h1_orig)

    print("\nMostrando coordenadas subpíxel corregidas en yosemite1.jpg...\n")
    subpix1 = refine_corners(im1, h1_keypoints)
    print_multiple_im(subpix1)

    print("\nDetectando puntos Harris en yosemite2.jpg...\n")
    h2_scale, h2_orig, h2_keypoints = harris_detection(im2, 3)
    print_im(format_pyramid(h2_scale))
    print_im(h2_orig)

    print("\nMostrando coordenadas subpíxel corregidas en yosemite2.jpg...\n")
    subpix2 = refine_corners(im2, h2_keypoints)
    print_multiple_im(subpix2)

#
# EJERCICIO 2: CORRESPONDENCIAS
#

def akaze_descriptor(im):
    """Devuelve los keypoints y descriptores AKAZE de una imagen."""

    return cv2.AKAZE_create().detectAndCompute(im, None)

def matches_bruteforce(desc1, desc2):
    """Devuelve los matches entre los descriptores de dos imágenes por el
       método de fuerza bruta con crossCheck"""

    # Creamos el objeto BFMatcher con crossCheck
    bf = cv2.BFMatcher_create(normType = cv2.NORM_HAMMING, crossCheck = True)

    # Calculamos los matches entre los descriptores de las imágenes
    matches = bf.match(desc1, desc2)

    return matches

def matches_lowe_2nn(desc1, desc2):
    """Devuelve los matches entre los descriptores de dos imágenes por el
       método de Lowe-Average-2NN"""

    # Creamos el objeto BFMatcher
    bf = cv2.BFMatcher_create(normType = cv2.NORM_HAMMING)

    # Calculamos los 2 mejores matches entre los descriptores de las imágenes
    matches = bf.knnMatch(desc1, desc2, k = 2)

    # Descartamos correspondencias ambiguas según el criterio de Lowe
    selected = []
    for m1, m2 in matches:
        if m1.distance < 0.75 * m2.distance:
            selected.append(m1)

    return selected

def get_matches(im1, im2):
    """Devuelve dos imágenes con las correspondencias entre los keypoints
       extraídos de 'im1' e 'im2' usando el descriptor AKAZE: una con el método
       BruteForce + crossCheck y otra con Lowe-Average-2NN."""

    # Obtenemos keypoints y sus descriptores con AKAZE
    kp1, desc1 = akaze_descriptor(im1)
    kp2, desc2 = akaze_descriptor(im2)

    # Obtenemos 100 matches aleatorios con BruteForce + crossCheck
    matches_bf = matches_bruteforce(desc1, desc2)
    n = min(100, len(matches_bf))
    matches_bf = random.sample(matches_bf, n)

    # Dibujamos las corresondencias
    im_matches_bf = cv2.drawMatches(im1, kp1, im2, kp2, matches_bf, None,
                                    flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # Obtenemos 100 matches aleatorios con Lowe-Average-2NN
    matches_2nn = matches_lowe_2nn(desc1, desc2)
    n = min(100, len(matches_2nn))
    matches_2nn = random.sample(matches_2nn, n)

    # Dibujamos las corresondencias
    im_matches_2nn = cv2.drawMatches(im1, kp1, im2, kp2, matches_2nn, None,
                                    flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    return im_matches_bf.astype(np.float32), im_matches_2nn.astype(np.float32)


def ex2():
    """Ejemplo de ejecución del ejercicio 2."""

    im1 = read_im(PATH + "yosemite1.jpg").astype(np.uint8)
    im2 = read_im(PATH + "yosemite2.jpg").astype(np.uint8)

    im_matches_bf, im_matches_2nn = get_matches(im1, im2)

    print("Correspondencias con BruteForce + crossCheck en yosemite...\n")
    print_im(im_matches_bf)

    print("\nCorrespondencias con Lowe-Average-2NN en yosemite...\n")
    print_im(im_matches_2nn)

    im3 = read_im(PATH + "mosaico002.jpg").astype(np.uint8)
    im4 = read_im(PATH + "mosaico003.jpg").astype(np.uint8)

    im_matches_bf, im_matches_2nn = get_matches(im3, im4)

    print("\nCorrespondencias con BruteForce + crossCheck en mosaico...\n")
    print_im(im_matches_bf)

    print("\nCorrespondencias con Lowe-Average-2NN en mosaico...\n")
    print_im(im_matches_2nn)

#
# EJERCICIO 3: MOSAICO CON 2 IMÁGENES
#

def ex3():
    """Ejemplo de ejecución del ejercicio 3."""

    im1 = read_im(PATH + "yosemite1.jpg", 1)
    im2 = read_im(PATH + "yosemite2.jpg", 1)

    # Definimos un canvas suponiendo que las imágenes tienen la misma altura
    h, w = im1.shape[0], 940
    canvas = np.zeros((h, w, 3), dtype = np.float32)

    # La homografía que lleva la primera imagen al mosaico es la identidad
    canvas[:im1.shape[0], :im1.shape[1]] = im1

    # Calculamos la homografía de la segunda imagen a la primera
    kp1, desc1 = akaze_descriptor(im1)
    kp2, desc2 = akaze_descriptor(im2)

    matches = matches_lowe_2nn(desc2, desc1)

    query = np.array([kp2[match.queryIdx].pt for match in matches])
    train = np.array([kp1[match.trainIdx].pt for match in matches])

    # Coincide en este caso con la homografía que lleva la segunda imagen al mosaico
    H21 = cv2.findHomography(query, train, cv2.RANSAC)[0]

    # Trasladamos la segunda imagen al mosaico
    canvas = cv2.warpPerspective(im2, H21, (w, h), dst = canvas, borderMode = cv2.BORDER_TRANSPARENT)

    # Mostramos el mosaico
    print_im(canvas)

#
# EJERCICIO 4: MOSAICO CON N IMÁGENES
#

def ex4(ims, w):
    """Ejemplo de ejecución del ejercicio 4.
        - ims: imágenes para construir el mosaico.
        - w: anchura del mosaico."""

    central = len(ims) // 2

    # Definimos un canvas suponiendo que las imágenes tienen la misma altura
    h = ims[0].shape[0] + 20
    canvas = np.zeros((h, w, 3), dtype = np.float32)

    # Calculamos la homografía (traslación) que lleva la imagen central al mosaico
    tx = (w - ims[central].shape[1]) / 2
    ty = (h - ims[central].shape[0]) / 2
    H0 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Calculamos las homografías entre cada dos imágenes
    homographies = []
    for i in range(len(ims)):
        if i != central:
            j = i + 1 if i < central else i - 1

            kp1, desc1 = akaze_descriptor(ims[i])
            kp2, desc2 = akaze_descriptor(ims[j])

            matches = matches_lowe_2nn(desc1, desc2)

            query = np.array([kp1[match.queryIdx].pt for match in matches])
            train = np.array([kp2[match.trainIdx].pt for match in matches])

            homographies.append(cv2.findHomography(query, train, cv2.RANSAC)[0])

        else: # No se usa la posición central
            homographies.append(np.array([]))

    # Trasladamos cada imagen al mosaico
    for i in range(len(ims)):
        H = H0
        
        for j in range(abs(i - central)):
            k = central - j - 1 if i < central else central + j + 1
            H = H @ homographies[k]

        canvas = cv2.warpPerspective(ims[i], H, (w, h), dst = canvas, borderMode = cv2.BORDER_TRANSPARENT)

    # Mostramos el mosaico
    print_im(canvas)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta la práctica 3 paso a paso. Cada apartado es una llamada a una función."""

    print("--- EJERCICIO 1: DETECTOR DE HARRIS ---\n")
    #ex1()

    print("\n--- EJERCICIO 2: CORRESPONDENCIAS ENTRE KEYPOINTS ---\n")
    #ex2()

    print("\n--- EJERCICIO 3: MOSAICO CON 2 IMÁGENES ---\n")
    #ex3()

    print("\n--- EJERCICIO 4: MOSAICO CON N IMÁGENES ---\n")

    print("Yosemite 1-4...\n")
    ims = [read_im(PATH + "{}{}.jpg".format("yosemite", i), 1) for i in range(1, 5)]
    ex4(ims, 1530)

    print("\nYosemite 5-7...\n")
    ims = [read_im(PATH + "{}{}.jpg".format("yosemite", i), 1) for i in range(5, 8)]
    ex4(ims, 1150)

if __name__ == "__main__":
  main()
