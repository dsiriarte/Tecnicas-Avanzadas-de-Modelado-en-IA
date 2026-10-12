"""Regresión del cierre y cancelación. Fuentes y GUI simuladas; no mide FPS."""
import json,threading,hashlib
from unittest.mock import patch
import numpy as np
import pipeline_a4 as p

class Fuente:
    def __init__(self,*args):self.i=0
    def isOpened(self):return True
    def get(self,key):return 6 if key==p.cv2.CAP_PROP_FRAME_COUNT else 30
    def read(self):
        if self.i>=6:return False,None
        self.i+=1
        return True,np.full((400,640,3),120,dtype=np.uint8)
    def release(self):pass

class FuenteFallida(Fuente):
    def isOpened(self):return False

def run():
    checks=[]
    fake_motor=lambda *args:lambda x:np.array([1.,0.,0.],dtype=np.float32)
    with patch.object(p,'motor',fake_motor):
        for size in [1,2]:
            with patch.object(p.cv2,'VideoCapture',Fuente):
                summary,frames,_=p.benchmark(None,'hilos_grafo',queue_size=size,save_example=False)
                assert summary['sin_perdidas'] and not summary['cancelado'] and summary['frames']==6
                assert frames.id.tolist()==list(range(6))
            checks.append(f'Fuente completa, cola {size}: 6 ids en orden y finalización completa')
            with patch.object(p.cv2,'VideoCapture',Fuente),patch.object(p.cv2,'namedWindow'),patch.object(p.cv2,'imshow'),patch.object(p.cv2,'waitKey',return_value=ord('q')),patch.object(p.cv2,'destroyAllWindows'):
                summary,frames,_=p.benchmark(None,'hilos_grafo',display=True,queue_size=size,save_example=False)
                assert summary['cancelado'] and not summary['sin_perdidas'] and len(frames)==1
            checks.append(f'Cancelación simulada, cola {size}: no se informa como video completo')
            with patch.object(p.cv2,'VideoCapture',FuenteFallida):
                for _ in range(20):
                    try:p.benchmark(None,'hilos_grafo',queue_size=size,save_example=False)
                    except RuntimeError as e:assert 'FileNotFoundError' in str(e)
                    else:raise AssertionError('La fuente fallida no produjo error')
            checks.append(f'Fuente fallida, cola {size}: 20 cierres correctos')
        assert not [t for t in threading.enumerate() if t.name in ['captura','inferencia','recursos']]
    result={'comprobaciones':checks,'hilos_residuales':0,'nota':'Pruebas funcionales con fuente, motor y GUI simulados; no son mediciones de rendimiento.',
            'sha256_version_medida':hashlib.sha256((p.RES/'pipeline_version_medida.txt').read_bytes()).hexdigest(),
            'sha256_version_revisada':hashlib.sha256((p.ROOT/'src/pipeline_a4.py').read_bytes()).hexdigest(),
            'cambio':'Lectura atómica de errores y comprobación de cancelación/conteo total. Pesos y etapas de procesamiento sin cambios.'}
    (p.RES/'revision_concurrencia.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':run()
