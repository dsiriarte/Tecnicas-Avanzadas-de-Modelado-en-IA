"""Equivalencia numérica y finalización segura ante un video inexistente."""
from pathlib import Path
import json,time,threading
import numpy as np
from pipeline_a4 import ROOT,RES,motor,benchmark,tf,cv2

def run():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1);cv2.setNumThreads(1)
    model=tf.keras.models.load_model(ROOT/'modelo/finetuning_s42.keras')
    eager=motor(model,False);graph=motor(model,True);results=[]
    for part in ['test','externo']:
        a=np.load(ROOT/f'datos/{part}_128.npz');p=[];q=[]
        for x in a['X']:
            x=x.astype('float32')[None];p.append(eager(x));q.append(graph(x))
        p=tf.nn.softmax(p).numpy();q=tf.nn.softmax(q).numpy()
        changes=int(np.sum(p.argmax(1)!=q.argmax(1)))
        assert changes==0
        results.append({'particion':part,'n':len(p),'cambios_clase':changes,
                        'max_delta_prob':float(np.max(np.abs(p-q))),
                        'accuracy_eager':float(np.mean(p.argmax(1)==a['y'])),
                        'accuracy_grafo':float(np.mean(q.argmax(1)==a['y']))})
    start=time.perf_counter();error=None
    try:benchmark(model,'hilos_grafo',video=Path('/tmp/u3_video_que_no_existe.avi'),save_example=False)
    except RuntimeError as e:error=str(e)
    assert error and 'FileNotFoundError' in error
    active=[t.name for t in threading.enumerate() if t.name in ['captura','inferencia','recursos']]
    assert not active
    result={'equivalencia':results,'error_fuente_manejado':True,'hilos_residuales':active,
            'segundos_fallo_con_calentamiento':time.perf_counter()-start,
            'gpu_visible':[d.name for d in tf.config.list_physical_devices('GPU')]}
    (RES/'verificacion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
