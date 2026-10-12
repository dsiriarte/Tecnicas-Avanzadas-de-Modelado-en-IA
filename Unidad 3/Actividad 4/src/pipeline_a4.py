"""Pipeline medible: captura, inferencia y presentación con identidades de cuadro.

La GUI usa una instalación aislada de OpenCV. Las pruebas sin --display no
se interpretan como FPS de visualización. No se descartan cuadros.
"""
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
if '--display' in sys.argv and (ROOT/'.gui-packages').exists():
    sys.path.insert(0,str(ROOT/'.gui-packages'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))
os.environ.setdefault('KERAS_HOME',str(ROOT/'.keras-cache'))
import argparse, json, queue, threading, time, traceback
import cv2
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

RES=ROOT/'resultados'; FIG=ROOT/'figuras'
NAMES=['Papel','Piedra','Tijera']
ROI=(359,172,484,359)
VIDEO=ROOT/'datos'/'rps_demo.avi'

def letterbox(rgb,size=128):
    h,w=rgb.shape[:2];scale=size/max(h,w)
    small=cv2.resize(rgb,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    out=np.full((size,size,3),127,dtype='uint8')
    y=(size-small.shape[0])//2;x=(size-small.shape[1])//2
    out[y:y+small.shape[0],x:x+small.shape[1]]=small
    return out

def preparar_frame(frame,roi=ROI):
    x1,y1,x2,y2=roi
    crop=frame[y1:y2,x1:x2]
    if not crop.size:raise ValueError('ROI fuera del cuadro')
    crop=cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE)
    rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    r,g,b=cv2.split(rgb.astype('float32'))
    green=(g>1.15*r)&(g>1.15*b)&(g>45)
    active=float((~green).mean())>.12
    return letterbox(rgb).astype('float32')[None],active

def motor(model,graph):
    if graph:
        infer=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec((1,128,128,3),tf.float32)])
    else:
        infer=lambda x:model(x,training=False)
    def call(x):
        return infer(tf.convert_to_tensor(x)).numpy()[0]
    # Calentamiento excluido del cronómetro; materialización sincroniza el resultado.
    x=np.full((1,128,128,3),127,dtype='float32')
    for _ in range(30):call(x)
    return call

def dibujar(frame,logits,active,mode,frame_id):
    out=frame.copy()
    score=np.exp(logits-logits.max());score/=score.sum();pred=int(score.argmax())
    x1,y1,x2,y2=ROI
    cv2.rectangle(out,(x1,y1),(x2,y2),(0,220,255),2)
    cv2.rectangle(out,(0,0),(640,51),(22,26,38),-1)
    label=f'{NAMES[pred]} {score[pred]:.3f}' if active else 'Sin gesto (regla de fondo)'
    cv2.putText(out,f'{mode} | cuadro {frame_id}',(10,18),cv2.FONT_HERSHEY_SIMPLEX,.46,(230,230,230),1,cv2.LINE_AA)
    cv2.putText(out,label,(10,42),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,255),1,cv2.LINE_AA)
    return out,pred,float(score.max())

def benchmark(model,mode,rep=0,display=False,loops=1,pace=False,video=VIDEO,save_example=True,pace_fps=None,queue_size=2):
    threaded=mode.startswith('hilos');graph=mode.endswith('grafo')
    infer=motor(model,graph)
    if queue_size<1:raise ValueError('La capacidad de cola debe ser positiva')
    stop=threading.Event();errors=queue.Queue();capture_q=queue.Queue(maxsize=queue_size);result_q=queue.Queue(maxsize=queue_size)
    process=psutil.Process();resources=[];records=[];start=0.;last_display=0.;expected_frames=None;cancelled=False
    def sample():
        process.cpu_percent(None)
        while not stop.wait(.1):
            resources.append({'s':time.perf_counter()-start,'cpu_pct_un_nucleo':process.cpu_percent(None),'rss_mb':process.memory_info().rss/2**20})
    def put(q,obj):
        while not stop.is_set():
            try:q.put(obj,timeout=.1);return True
            except queue.Full:pass
        return False
    def lanzar_error():
        # Una sola operación: dos consumidores no pueden bloquearse por leer el mismo error.
        try:message=errors.get_nowait()
        except queue.Empty:return
        raise RuntimeError(message)
    def get(q):
        while not stop.is_set():
            lanzar_error()
            try:return q.get(timeout=.1)
            except queue.Empty:pass
        return None
    def frames():
        nonlocal expected_frames
        fid=0
        for loop in range(loops):
            cap=cv2.VideoCapture(str(video))
            if not cap.isOpened():raise FileNotFoundError(video)
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            expected_frames=count*loops if count>0 else None
            fps=pace_fps or cap.get(cv2.CAP_PROP_FPS);frame_in_loop=0
            try:
                while not stop.is_set():
                    if pace:
                        wait=start+fid/fps-time.perf_counter()
                        if wait>0:stop.wait(wait)
                    t0=time.perf_counter();ok,frame=cap.read();t1=time.perf_counter()
                    if not ok:break
                    yield {'id':fid,'loop':loop,'frame_video':frame_in_loop,'frame':frame,'t0':t0,'lectura_ms':1000*(t1-t0),'t_captura':t1}
                    fid+=1;frame_in_loop+=1
            finally:cap.release()
    def calculate(item):
        a=time.perf_counter();x,active=preparar_frame(item['frame']);b=time.perf_counter()
        logits=infer(x);c=time.perf_counter()
        item.update(logits=logits,active=active,pre_ms=1000*(b-a),infer_ms=1000*(c-b),
                    cola_captura_ms=1000*(a-item['t_captura']),t_inferencia=c)
        return item
    def capture():
        try:
            for item in frames():
                if not put(capture_q,item):break
            put(capture_q,None)
        except BaseException:errors.put(traceback.format_exc());put(capture_q,None)
    def worker():
        try:
            while not stop.is_set():
                item=get(capture_q)
                if item is None:break
                if not put(result_q,calculate(item)):break
            put(result_q,None)
        except BaseException:errors.put(traceback.format_exc());put(result_q,None)
    def present(item):
        nonlocal last_display
        a=time.perf_counter();out,pred,conf=dibujar(item['frame'],item['logits'],item['active'],mode,item['id']);b=time.perf_counter()
        if display:
            cv2.imshow('U3 - Reconocimiento de gestos',out)
            if cv2.waitKey(1)&0xff==ord('q'):stop.set()
        c=time.perf_counter()
        record={'id':item['id'],'loop':item['loop'],'frame_video':item['frame_video'],'mode':mode,'rep':rep,
                'lectura_ms':item['lectura_ms'],'pre_ms':item['pre_ms'],'infer_ms':item['infer_ms'],
                'cola_captura_ms':item['cola_captura_ms'],'cola_salida_ms':1000*(a-item['t_inferencia']),
                'dibujo_ms':1000*(b-a),'presentacion_ms':1000*(c-b),'latencia_ms':1000*(c-item['t0']),
                'intervalo_ms':1000*(c-last_display) if last_display else np.nan,
                'tiempo_s':c-start,'pred':pred,'confianza':conf,'gesto_visible_regla':item['active']}
        records.append(record);last_display=c
        if save_example and rep==0 and item['id']==452:cv2.imwrite(str(FIG/f'ejemplo_{mode}.png'),out)
    if display:
        cv2.namedWindow('U3 - Reconocimiento de gestos',cv2.WINDOW_AUTOSIZE)
    start=time.perf_counter();sampler=threading.Thread(target=sample,name='recursos',daemon=True);sampler.start()
    threads=[]
    cpu_start=process.cpu_times()
    try:
        if threaded:
            threads=[threading.Thread(target=capture,name='captura'),threading.Thread(target=worker,name='inferencia')]
            for t in threads:t.start()
            while not stop.is_set():
                item=get(result_q)
                if item is None:break
                present(item)
        else:
            for item in frames():
                present(calculate(item))
                if stop.is_set():break
        lanzar_error()
    finally:
        elapsed=time.perf_counter()-start;cancelled=stop.is_set();stop.set()
        for t in threads:t.join(timeout=5)
        sampler.join(timeout=2)
        if any(t.is_alive() for t in threads):raise RuntimeError('Un hilo no terminó')
        if display:cv2.destroyAllWindows()
    cpu_end=process.cpu_times()
    df=pd.DataFrame(records)
    if df.empty:raise RuntimeError('No se procesaron cuadros')
    assert df.id.tolist()==list(range(len(df))), 'Pérdida, duplicación o reordenamiento de cuadros'
    # El FPS usa cuadros presentados / tiempo total; no 1/media(inferencia).
    windows=np.array([100/(df.tiempo_s.iloc[i+100]-df.tiempo_s.iloc[i]) for i in range(0,len(df)-100,100)])
    summary={'mode':mode,'rep':rep,'display':display,'pace':pace,'pace_fps':pace_fps,'frames':len(df),'segundos':elapsed,'fps':len(df)/elapsed,
             'latencia_media_ms':float(df.latencia_ms.mean()),'latencia_p50_ms':float(df.latencia_ms.quantile(.5)),
             'latencia_p95_ms':float(df.latencia_ms.quantile(.95)),'latencia_p99_ms':float(df.latencia_ms.quantile(.99)),
             'infer_media_ms':float(df.infer_ms.mean()),'infer_p95_ms':float(df.infer_ms.quantile(.95)),
             'fps_ventana_min':float(windows.min()) if len(windows) else None,'fps_ventana_p05':float(np.quantile(windows,.05)) if len(windows) else None,
             'ventanas_mayor60_pct':float(100*np.mean(windows>60)) if len(windows) else None,
             'cpu_equivalente_pct':100*((cpu_end.user+cpu_end.system)-(cpu_start.user+cpu_start.system))/elapsed,
             'rss_max_mb':max((r['rss_mb'] for r in resources),default=process.memory_info().rss/2**20),
             'sin_perdidas':bool(not cancelled and expected_frames is not None and len(df)==expected_frames),
             'cancelado':cancelled,'cuadros_esperados':expected_frames,'cola_max':queue_size,'opencv':cv2.__version__}
    return summary,df,pd.DataFrame(resources)

def main():
    p=argparse.ArgumentParser();p.add_argument('--display',action='store_true');p.add_argument('--reps',type=int,default=3)
    p.add_argument('--loops',type=int,default=1);p.add_argument('--pace',action='store_true');p.add_argument('--pace-fps',type=float);p.add_argument('--tag',default='benchmark')
    p.add_argument('--queue-size',type=int,default=2)
    p.add_argument('--modes',nargs='+',default=['secuencial_eager','hilos_eager','secuencial_grafo','hilos_grafo'])
    args=p.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1);cv2.setNumThreads(1)
    config=json.loads((ROOT/'modelo/config_A3.json').read_text())
    model_path=ROOT/'modelo'/f'{config["elegido"]}_s42.keras'
    model=tf.keras.models.load_model(model_path)
    summaries=[]
    for rep in range(args.reps):
        # Orden alternado para reducir sesgo por calentamiento y orden de ejecución.
        modes=args.modes if rep%2==0 else args.modes[::-1]
        for mode in modes:
            summary,df,resources=benchmark(model,mode,rep,args.display,args.loops,args.pace,pace_fps=args.pace_fps,queue_size=args.queue_size)
            stem=f'{args.tag}_{mode}_r{rep}'
            df.to_csv(RES/f'{stem}_frames.csv',index=False);resources.to_csv(RES/f'{stem}_recursos.csv',index=False)
            summaries.append(summary);pd.DataFrame(summaries).to_csv(RES/f'{args.tag}_resumen.csv',index=False)
            print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
