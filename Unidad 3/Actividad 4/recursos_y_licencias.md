# Recursos públicos y atribuciones

Autores del trabajo: David Santiago Iriarte Zamora y Jose Adrian Trochez Cuellar · Grupo 12.

## Fotografías reales de gestos

- Autor: DrGFreeman (Julien de la Bruère-Terreault).
- Fuente: https://github.com/DrGFreeman/rps-cv-data-science
- Descarga enlazada por el autor: https://www.dropbox.com/s/sr1iui3zfdk0o53/rps-cv-image.zip?dl=1
- Licencia: **Creative Commons Atribución-CompartirIgual 4.0 Internacional**, https://creativecommons.org/licenses/by-sa/4.0/
- 2.188 fotografías: papel 712, piedra 726, tijera 750, según los archivos descargados. Estos conteos por clase difieren ligeramente del README de origen.
- Modificaciones: organización en particiones, conversión RGB, redimensionamiento con proporción y relleno a 128 × 128, y selección de ejemplos anotados en figuras. Los datos NPZ derivados y reproducciones/adaptaciones de estas imágenes se comparten bajo la misma CC BY-SA 4.0, manteniendo esta atribución. No atribuimos a nuestros autores las fotografías originales.

## Imágenes CGI externas

- Autor: Laurence Moroney.
- Fuente y licencia declarada por el autor: https://laurencemoroney.com/datasets.html
- Descarga: https://storage.googleapis.com/learning-datasets/rps-test-set.zip
- Licencia: **Creative Commons Atribución 2.0 Genérica**, https://creativecommons.org/licenses/by/2.0/
- 372 imágenes, 124 por clase. Solo evaluación externa, sin entrenamiento ni selección.
- Modificaciones: conversión a tensor RGB 128 × 128 y figuras anotadas. Se mantiene atribución y enlace a la licencia.

## Demostración de video

- Autor: Julien de la Bruère-Terreault (DrGFreeman).
- Fuente: https://github.com/DrGFreeman/rps-cv
- Revisión fija: `15ca5d41b97b9f69b6f90cd52d3664bd947070a6`.
- Archivo: `img/doc/rps.gif`. Se conserva la licencia **MIT del repositorio** en `datos/LICENSE_demo.txt`, incluido el aviso de copyright. No se identificó una licencia separada para este archivo de documentación.
- Modificaciones: conversión de los 904 cuadros a AVI MJPG, cadencia fija 14,2857 FPS; ROI del panel del jugador, rotación, anotaciones y reproducción acelerada para medición. El GIF original dura 60,27 s; el AVI dura 63,28 s. La interfaz y marcadores del juego pertenecen al material original.
- El video no aporta etiquetas independientes por cuadro. No lo tratamos como una prueba de generalización.

## Pesos y documentación técnica

- MobileNetV2 (Sandler et al., 2018): https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html
- Keras MobileNet: https://keras.io/api/applications/mobilenet/
- Pesos oficiales ImageNet, α=0,35, entrada 128, sin clasificador: https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_128_no_top.h5
- TensorFlow, transferencia: https://www.tensorflow.org/tutorials/images/transfer_learning
- PyTorch, transferencia (recurso de la consigna, contraste conceptual): https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- TensorFlow, grafo: https://www.tensorflow.org/guide/function
- OpenCV, medición: https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html
- Python, hilos: https://docs.python.org/3.12/library/threading.html
- Python, colas: https://docs.python.org/3.12/library/queue.html
- Ultralytics, benchmark (recurso de la consigna; no usamos YOLO): https://docs.ultralytics.com/modes/benchmark/

`datos/fuentes.json` registra URLs, autores, licencias, tamaños y SHA-256 de las descargas. `resultados/config_A3.json` (A3) y `modelo/config_A3.json` (A4) registran pesos iniciales y partición; `resultados/analisis.json` en A4 registra el modelo y video utilizados. Las referencias APA también se encuentran en los informes y notebooks. Las fuentes respaldan el método; las cifras de desempeño provienen de nuestras ejecuciones locales.
