"""Resume mediciones crudas; no vuelve a ejecutar el benchmark visual."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RES=ROOT/'resultados';FIG=ROOT/'figuras'
ORDER=['secuencial_eager','hilos_eager','secuencial_grafo','hilos_grafo']
LABELS=['Secuencial\neager','Hilos\neager','Secuencial\ngrafo','Hilos\ngrafo']
COLORS=['#929daa','#d99b3d','#4b89a9','#248775']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})

def run():
    original=pd.read_csv(RES/'visual_resumen.csv')
    small=pd.read_csv(RES/'cola1_resumen.csv')
    sustained=pd.read_csv(RES/'sostenido60_resumen.csv')
    original['configuracion']=original['mode']+' · cola 2'
    small['configuracion']='hilos_grafo · cola 1'
    sustained['configuracion']='hilos_grafo · entrada 60'
    allruns=pd.concat([original,small,sustained],ignore_index=True)
    measures=['fps','latencia_media_ms','latencia_p95_ms','infer_media_ms','cpu_equivalente_pct','rss_max_mb','fps_ventana_min']
    summary=allruns.groupby('configuracion',sort=False)[measures].agg(['mean','std'])
    summary.to_csv(RES/'comparacion.csv')
    base=pd.read_csv(RES/'visual_secuencial_eager_r0_frames.csv')
    equivalence=[]
    for path in sorted(RES.glob('*_frames.csv')):
        d=pd.read_csv(path);reference=base.iloc[d.frame_video.to_numpy()]
        assert d.id.tolist()==list(range(len(d))),path
        assert len(d)%904==0,path
        changed=int(np.sum(d.pred.to_numpy()!=reference.pred.to_numpy()))
        equivalence.append({'archivo':path.name,'cuadros':len(d),'clases_distintas':changed,
                            'max_delta_confianza':float(np.max(np.abs(d.confianza.to_numpy()-reference.confianza.to_numpy())))})
    pd.DataFrame(equivalence).to_csv(RES/'equivalencia_video.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    groups=original.groupby('mode')
    means=[groups.get_group(m).fps.mean() for m in ORDER];std=[groups.get_group(m).fps.std() for m in ORDER]
    axs[0].bar(LABELS,means,yerr=std,color=COLORS,capsize=3)
    axs[0].axhline(60,ls='--',color='#a23535',lw=1);axs[0].set_ylabel('Cuadros presentados / segundo')
    axs[0].set_title('Tres repeticiones · media ± DE')
    for i,y in enumerate(means):axs[0].text(i,y+2,f'{y:.1f}',ha='center')
    stages=['lectura_ms','pre_ms','infer_ms','dibujo_ms','presentacion_ms'];stage_names=['Lectura','Preproc.','Inferencia','Dibujo','Presentación']
    timings=[]
    for mode in ORDER:
        d=pd.concat([pd.read_csv(RES/f'visual_{mode}_r{r}_frames.csv') for r in range(3)])
        timings.append([d[s].mean() for s in stages])
    pd.DataFrame(timings,index=ORDER,columns=stages).to_csv(RES/'tiempos_etapas.csv')
    bottom=np.zeros(4)
    for j,(name,color) in enumerate(zip(stage_names,['#adc6d9','#b6c5b0','#285879','#d7a55a','#78669b'])):
        values=np.array(timings)[:,j];axs[1].bar(LABELS,values,bottom=bottom,label=name,color=color);bottom+=values
    axs[1].set_ylabel('Trabajo medio por cuadro (ms)');axs[1].set_title('Etapas activas; esperas de cola excluidas')
    axs[1].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(FIG/'rendimiento.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.2))
    variants=[('visual','secuencial_eager','Secuencial'),('visual','hilos_grafo','Hilos · cola 2'),('cola1','hilos_grafo','Hilos · cola 1')]
    for tag,mode,label in variants:
        d=pd.read_csv(RES/f'{tag}_{mode}_r0_frames.csv')
        t=d.tiempo_s.to_numpy();w=[100/(t[i+100]-t[i]) for i in range(0,len(d)-100,100)]
        axs[0].plot(np.arange(len(w))+1,w,marker='o',ms=3,label=label)
        a=np.sort(d.latencia_ms);axs[1].plot(a,np.arange(1,len(a)+1)/len(a),label=label)
    axs[0].axhline(60,color='#a23535',ls='--');axs[0].set(xlabel='Ventana no superpuesta de 100 intervalos',ylabel='FPS',title='Estabilidad en la primera repetición')
    axs[1].set(xlabel='Latencia de lectura a presentación (ms)',ylabel='Proporción acumulada',title='Mayor caudal no implica menor demora')
    axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'latencia_estabilidad.png',dpi=180);plt.close(fig)
    d=pd.read_csv(RES/'sostenido60_hilos_grafo_r0_frames.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,2.9))
    axs[0].plot(d.tiempo_s,d.latencia_ms,lw=.5,color='#248775');axs[0].axhline(1000/60,ls='--',color='#a23535',label='16,67 ms')
    axs[0].set(xlabel='Tiempo (s)',ylabel='Latencia (ms)',title='Entrada controlada de 60 FPS');axs[0].legend(fontsize=8)
    resource=pd.read_csv(RES/'sostenido60_hilos_grafo_r0_recursos.csv')
    axs[1].plot(resource.s,resource.rss_mb,color='#285879');axs[1].set(xlabel='Tiempo (s)',ylabel='Memoria residente (MiB)',title='Memoria acotada durante cuatro ciclos')
    fig.tight_layout();fig.savefig(FIG/'sostenido.png',dpi=180);plt.close(fig)
    result={'speedup_cola1':float(small.fps.mean()/original.query("mode == 'secuencial_eager'").fps.mean()),
            'min_ventana_cola1':float(small.fps_ventana_min.min()),'clases_distintas_video':sum(x['clases_distintas'] for x in equivalence),
            'max_delta_confianza_video':max(x['max_delta_confianza'] for x in equivalence),
            'cuadros_comparados':sum(x['cuadros'] for x in equivalence),
            'sha256_video':hashlib.sha256((ROOT/'datos/rps_demo.avi').read_bytes()).hexdigest(),
            'sha256_modelo':hashlib.sha256((ROOT/'modelo/finetuning_s42.keras').read_bytes()).hexdigest()}
    (RES/'analisis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(summary.round(3).to_string());print(result)

if __name__=='__main__':run()
