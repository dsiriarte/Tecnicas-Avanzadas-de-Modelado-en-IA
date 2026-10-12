"""A3: transferencia y ajuste fino. Selección en validación antes de abrir prueba."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
os.environ.setdefault('KERAS_HOME',str(ROOT/'.keras-cache'))
import argparse, hashlib, json, platform, time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA=ROOT/'datos'; ACT=ROOT; RES=ACT/'resultados'; MODELS=ACT/'modelos'
CLASES=['paper','rock','scissors']
PARAMS={'arquitectura':'MobileNetV2','alpha':.35,'imagen':128,'pesos':'ImageNet',
        'batch':32,'semillas':[42,7,2026],'epocas_congelada':10,'epocas_finetuning':12,
        'lr_congelada':.001,'lr_finetuning':.0001,'paciencia':4,'fine_desde':'block_13_expand',
        'seleccion':'Mayor F1-macro medio en validación; empate a 0.001 favorece congelada.',
        'aumentacion':{'flip':'horizontal','rotacion_fraccion':.05,'traslacion':.05,'zoom':.10,'contraste':.15,'brillo':.10}}

_CONFIGURADO=False
def configurar():
    global _CONFIGURADO
    if _CONFIGURADO:return
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    _CONFIGURADO=True

def cargar(part):
    a=np.load(DATA/f'{part}_128.npz')
    return a['X'],a['y']

def modelo_nuevo():
    base=tf.keras.applications.MobileNetV2(input_shape=(128,128,3),alpha=.35,include_top=False,weights=None)
    base.load_weights(DATA/'mobilenetv2_035_128_imagenet.weights.h5')
    base.trainable=False
    inp=tf.keras.Input((128,128,3),name='rgb_0_255')
    x=tf.keras.layers.Rescaling(1/127.5,offset=-1,name='normalizacion_imagenet')(inp)
    x=base(x,training=False)
    x=tf.keras.layers.GlobalAveragePooling2D(name='promedio_espacial')(x)
    x=tf.keras.layers.Dropout(.2,name='dropout')(x)
    out=tf.keras.layers.Dense(3,name='logits')(x)
    return tf.keras.Model(inp,out,name='clasificador_rps'),base

def datasets(seed):
    x,y=cargar('train'); xv,yv=cargar('val')
    L=tf.keras.layers
    aug=tf.keras.Sequential([L.RandomFlip('horizontal',seed=seed),L.RandomRotation(.05,fill_mode='reflect',seed=seed+1),
        L.RandomTranslation(.05,.05,fill_mode='reflect',seed=seed+2),L.RandomZoom(.1,fill_mode='reflect',seed=seed+3),
        L.RandomContrast(.15,seed=seed+4),L.RandomBrightness(.1,value_range=(0,255),seed=seed+5)])
    ds=tf.data.Dataset.from_tensor_slices((x,y)).shuffle(len(y),seed=seed,reshuffle_each_iteration=True).batch(PARAMS['batch'])
    ds=ds.map(lambda a,b:(aug(a,training=True),b),num_parallel_calls=1).prefetch(1)
    opts=tf.data.Options();opts.threading.private_threadpool_size=2;opts.experimental_deterministic=True
    return ds.with_options(opts),tf.data.Dataset.from_tensor_slices((xv,yv)).batch(64).with_options(opts)

class Registro(tf.keras.callbacks.Callback):
    def __init__(self,seed,stage):super().__init__();self.seed=seed;self.stage=stage;self.times=[]
    def on_epoch_begin(self,epoch,logs=None):self.start=time.perf_counter()
    def on_epoch_end(self,epoch,logs=None):
        self.times.append(time.perf_counter()-self.start)
        print(f'{self.stage} s{self.seed} ep{epoch+1} loss={logs["loss"]:.4f} val={logs["val_loss"]:.4f} acc={logs["val_accuracy"]:.4f} {self.times[-1]:.1f}s',flush=True)

def entrenar_etapa(model,train,val,seed,stage,epochs,lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    reg=Registro(seed,stage)
    start=time.perf_counter()
    hist=model.fit(train,validation_data=val,epochs=epochs,verbose=0,shuffle=False,callbacks=[reg,
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=PARAMS['paciencia'],min_delta=1e-4,restore_best_weights=True)])
    elapsed=time.perf_counter()-start
    model.save(MODELS/f'{stage}_s{seed}.keras')
    record={'history':hist.history,'epocas_s':reg.times,'total_s':elapsed,'mejor_epoca':int(np.argmin(hist.history['val_loss'])+1),
            'parametros':model.count_params(),'entrenables':int(sum(np.prod(v.shape) for v in model.trainable_weights)),
            'lr':lr,'seed':seed,'stage':stage}
    (RES/f'historia_{stage}_s{seed}.json').write_text(json.dumps(record,indent=2))
    return record

def probabilidades(model,x):
    return tf.nn.softmax(model.predict(x,batch_size=64,verbose=0)).numpy()

def run(seeds=None,force=False):
    configurar();seeds=seeds or PARAMS['semillas']
    (RES/'protocolo_previo.json').write_text(json.dumps(PARAMS,indent=2,ensure_ascii=False))
    val_metrics=[]
    for seed in seeds:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        train,val=datasets(seed)
        if force or not (MODELS/f'congelada_s{seed}.keras').exists():
            model,base=modelo_nuevo()
            before=[v.numpy().copy() for v in base.weights]
            entrenar_etapa(model,train,val,seed,'congelada',PARAMS['epocas_congelada'],PARAMS['lr_congelada'])
            assert all(np.array_equal(a,b.numpy()) for a,b in zip(before,base.weights)), 'La base congelada cambió'
        model=tf.keras.models.load_model(MODELS/f'congelada_s{seed}.keras')
        if force or not (MODELS/f'finetuning_s{seed}.keras').exists():
            base=next(l for l in model.layers if isinstance(l,tf.keras.Model))
            base.trainable=True
            reached=False
            for layer in base.layers:
                if layer.name==PARAMS['fine_desde']:reached=True
                layer.trainable=reached and not isinstance(layer,tf.keras.layers.BatchNormalization)
            assert reached
            watched=base.get_layer('block_13_expand').get_weights()[0].copy()
            rec=entrenar_etapa(model,train,val,seed,'finetuning',PARAMS['epocas_finetuning'],PARAMS['lr_finetuning'])
            delta=float(np.linalg.norm(base.get_layer('block_13_expand').get_weights()[0]-watched))
            assert delta>0,'No hubo ajuste fino'
            (RES/f'control_finetuning_s{seed}.json').write_text(json.dumps({'cambio_L2_block13':delta,'batchnorm_entrenables':sum(l.trainable for l in base.layers if isinstance(l,tf.keras.layers.BatchNormalization))}))
        xv,yv=cargar('val')
        for stage in ['congelada','finetuning']:
            model=tf.keras.models.load_model(MODELS/f'{stage}_s{seed}.keras');p=probabilidades(model,xv)
            val_metrics.append({'stage':stage,'seed':seed,'f1_macro':float(f1_score(yv,p.argmax(1),average='macro')),'accuracy':float(accuracy_score(yv,p.argmax(1)))})
    vals=pd.DataFrame(val_metrics);vals.to_csv(RES/'seleccion_validacion.csv',index=False)
    means=vals.groupby('stage').f1_macro.mean()
    elegido='finetuning' if means['finetuning']>means['congelada']+.001 else 'congelada'
    config={'params':PARAMS,'semillas_ejecutadas':seeds,'elegido':elegido,'semilla_despliegue':42,
            'versiones':{'python':platform.python_version(),'tensorflow':tf.__version__,'keras':tf.keras.__version__,
            'numpy':np.__version__,'plataforma':platform.platform(),'cpu_logicos':os.cpu_count(),'dispositivo':'CPU'},
            'sha256_particiones':hashlib.sha256((RES/'particiones.csv').read_bytes()).hexdigest(),
            'sha256_pesos_preentrenados':hashlib.sha256((DATA/'mobilenetv2_035_128_imagenet.weights.h5').read_bytes()).hexdigest()}
    # Decisión persistida ANTES de evaluar test y externo; no se selecciona semilla por prueba.
    (RES/'config_A3.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    print('Seleccionado por validación:',elegido,flush=True)
    results=[];perclass=[]
    for seed in seeds:
        for stage in ['congelada','finetuning']:
            model=tf.keras.models.load_model(MODELS/f'{stage}_s{seed}.keras')
            for part in ['train','val','test','externo']:
                x,y=cargar(part);p=probabilidades(model,x);yp=p.argmax(1)
                np.savez_compressed(RES/f'pred_{stage}_s{seed}_{part}.npz',y=y,prob=p)
                results.append({'stage':stage,'seed':seed,'particion':part,'n':len(y),
                                'accuracy':accuracy_score(y,yp),'f1_macro':f1_score(y,yp,average='macro')})
                if part in ['test','externo']:
                    report=classification_report(y,yp,target_names=CLASES,output_dict=True,zero_division=0)
                    for c in CLASES:perclass.append({'stage':stage,'seed':seed,'particion':part,'clase':c,**report[c]})
    pd.DataFrame(results).to_csv(RES/'metricas.csv',index=False)
    pd.DataFrame(perclass).to_csv(RES/'metricas_por_clase.csv',index=False)
    print(pd.DataFrame(results).groupby(['stage','particion'])[['accuracy','f1_macro']].agg(['mean','std']).round(4),flush=True)
    return config

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=PARAMS['semillas'])
    parser.add_argument('--force',action='store_true',help='Reentrenar y reemplazar los modelos generados de estas semillas')
    args=parser.parse_args();run(args.seeds,args.force)
