- categorical_crossentropy is another term for multi-class log loss (http://wiki.fast.ai/index.php/Log_Loss)

def logloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -log(p)
  else:
    return -log(1 - p)

- Cortamos en 25 épocas porque se observa que el accuracy de validación comienza a bajar a partir de este número:

imágenes: loss y accuracy hasta 30 épocas.

- Batchnormalization already includes the addition of the bias term. So there is no need (and it makes no sense) to add another bias term in the convolution layer. Simply speaking BatchNorm shifts the activation by their mean values. Hence, any constant will be canceled out.
          bn = gamma * normalized(x) + bias
