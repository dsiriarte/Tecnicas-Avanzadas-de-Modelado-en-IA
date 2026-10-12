"""Tablas y figuras a partir de predicciones e historiales guardados, sin ajustar modelos."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score
RES=ROOT/'resultados';FIG=ROOT/'figuras'
NAMES=['Papel','Piedra','Tijera'];SEEDS=[42,7,2026]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})

def paired_bootstrap(delta,groups,B=2000):
    codes,groups=np.unique(groups,return_inverse=True)
    totals=np.bincount(groups,weights=delta);sizes=np.bincount(groups)
    rng=np.random.default_rng(2026);est=[]
    for _ in range(B):
        ids=rng.integers(len(codes),size=len(codes));est.append(totals[ids].sum()/sizes[ids].sum())
    return float(delta.mean()),*map(float,np.quantile(est,[.025,.975]))

def run():
    metrics=pd.read_csv(RES/'metricas.csv')
    summary=metrics.groupby(['stage','particion']).agg(accuracy=('accuracy','mean'),accuracy_sd=('accuracy','std'),f1=('f1_macro','mean'),f1_sd=('f1_macro','std')).reset_index()
    summary.to_csv(RES/'resumen_metricas.csv',index=False)
    intervals=[]
    for part in ['test','externo']:
        data=np.load(ROOT/f'datos/{part}_128.npz');y=data['y'];groups=data['groups']
        ac={}
        for stage in ['congelada','finetuning']:
            ac[stage]=np.mean([np.load(RES/f'pred_{stage}_s{s}_{part}.npz')['prob'].argmax(1)==y for s in SEEDS],axis=0)
        d,lo,hi=paired_bootstrap(ac['finetuning']-ac['congelada'],groups)
        intervals.append({'particion':part,'delta_accuracy':d,'IC95_inf':lo,'IC95_sup':hi,'grupos':len(np.unique(groups))})
    pd.DataFrame(intervals).to_csv(RES/'bootstrap.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(10,5.5))
    for j,stage in enumerate(['congelada','finetuning']):
        for s,c in zip(SEEDS,['#1f4e79','#dd8a22','#419574']):
            h=json.loads((RES/f'historia_{stage}_s{s}.json').read_text())['history'];ep=np.arange(1,len(h['loss'])+1)
            for i,metric in enumerate(['loss','accuracy']):
                axs[i,j].plot(ep,h[metric],color=c,ls='--',alpha=.8,label=f'Train s{s}')
                axs[i,j].plot(ep,h['val_'+metric],color=c,label=f'Val s{s}')
                axs[i,j].set_ylabel('Entropía cruzada' if i==0 else 'Exactitud');axs[i,j].set_xlabel('Época');axs[i,j].grid(alpha=.2)
        axs[0,j].set_title('Base congelada' if j==0 else 'Ajuste fino parcial')
    axs[1,1].legend(ncol=2,fontsize=7,loc='lower right');fig.tight_layout();fig.savefig(FIG/'curvas.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(10,3.1))
    cases=[('finetuning','test','Prueba real · ajuste fino'),('congelada','externo','CGI · base congelada'),('finetuning','externo','CGI · ajuste fino')]
    for ax,(stage,part,title) in zip(axs,cases):
        p=np.load(RES/f'pred_{stage}_s42_{part}.npz');m=confusion_matrix(p['y'],p['prob'].argmax(1),labels=[0,1,2]);norm=m/m.sum(1,keepdims=True)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{m[i,j]}\n{norm[i,j]:.0%}',ha='center',va='center',color='white' if norm[i,j]>.55 else 'black',fontsize=9)
        ax.set_xticks(range(3),NAMES);ax.set_yticks(range(3),NAMES);ax.set_xlabel('Predicción');ax.set_ylabel('Etiqueta');ax.set_title(title)
    fig.tight_layout();fig.savefig(FIG/'confusion.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(9,5.4))
    for row,part in enumerate(['test','externo']):
        data=np.load(ROOT/f'datos/{part}_128.npz');p=np.load(RES/f'pred_finetuning_s42_{part}.npz')['prob'];y=data['y'];yp=p.argmax(1)
        candidates=np.flatnonzero(yp==y) if row==0 else np.flatnonzero(yp!=y)
        chosen=[]
        for c in range(3):
            ids=[i for i in candidates if y[i]==c]
            if ids:chosen.append(sorted(ids,key=lambda i:-p[i].max())[0])
        for i in sorted(candidates,key=lambda i:-p[i].max()):
            if len(chosen)>=3:break
            if i not in chosen:chosen.append(i)
        for ax,idx in zip(axs[row],chosen):
            ax.imshow(data['X'][idx]);ax.axis('off');ax.set_title(f'{part}: {NAMES[y[idx]]}\nPred.: {NAMES[yp[idx]]} ({p[idx].max():.3f})')
    fig.tight_layout();fig.savefig(FIG/'aciertos_errores.png',dpi=180);plt.close(fig)
    # Desempeño por clase en promedio entre semillas, sin sumar soportes.
    pc=pd.read_csv(RES/'metricas_por_clase.csv')
    pc.groupby(['stage','particion','clase'])[['precision','recall','f1-score','support']].mean().to_csv(RES/'resumen_por_clase.csv')
    # Ancho de IC binomial: evitar interpretar 437/437 como garantía de perfección.
    from statsmodels.stats.proportion import proportion_confint
    n=len(np.load(ROOT/'datos/test_128.npz')['y'])
    lo,hi=proportion_confint(n,n,alpha=.05,method='wilson')
    train=np.load(ROOT/'datos/train_128.npz')['y'];test=np.load(ROOT/'datos/test_128.npz')['y']
    majority=int(np.bincount(train).argmax())
    result={'wilson_test_437_aciertos':[float(lo),float(hi)],'mayoritaria_accuracy':float(np.mean(test==majority)),
            'mayoritaria_f1':float(f1_score(test,np.full_like(test,majority),average='macro')),
            'limite_IC':'Wilson orientativo por imagen; no corrige dependencia por sujetos no identificados.',
            'bootstrap':intervals}
    (RES/'analisis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(summary.to_string(index=False));print(result)

if __name__=='__main__':run()
