"""Obtiene los pesos oficiales de Keras solo si aún no están en el proyecto."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('KERAS_HOME',str(ROOT/'.keras-cache'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')

def preparar():
    target=ROOT/'datos'/'mobilenetv2_035_128_imagenet.weights.h5'
    if not target.exists():
        import tensorflow as tf
        base=tf.keras.applications.MobileNetV2(input_shape=(128,128,3),alpha=.35,include_top=False,weights='imagenet')
        base.save_weights(target)
    return target

if __name__=='__main__':print(preparar())
