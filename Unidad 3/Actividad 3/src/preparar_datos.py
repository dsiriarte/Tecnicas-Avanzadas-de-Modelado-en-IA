"""Inventario, grupos visuales y partición fija antes del entrenamiento."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import hashlib, json
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASES = ['paper', 'rock', 'scissors']
NOMBRES = ['Papel', 'Piedra', 'Tijera']
DATA = ROOT / 'datos'
RES = ROOT / 'resultados'
FIG = ROOT / 'figuras'
SIZE = 128

def preparar():
    rows, mini, hashes = [], [], []
    for label, clase in enumerate(CLASES):
        for p in sorted((DATA / 'real' / clase).glob('*.png')):
            im = cv2.imread(str(p)); assert im is not None, p
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype('float32').ravel()
            x -= x.mean(); x /= max(np.linalg.norm(x), 1e-6)
            mini.append(x)
            tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            hashes.append(np.packbits(tiny[:, 1:] > tiny[:, :-1]).view('uint64')[0])
            rows.append(dict(path=str(p.relative_to(ROOT)), clase=clase, label=label,
                             alto=im.shape[0], ancho=im.shape[1],
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
    df = pd.DataFrame(rows)
    df.to_csv(RES / 'inventario.csv', index=False)
    # Grupos conservadores: copia exacta o dHash <= 4 y correlación >= 0.985.
    x = np.array(mini); h = np.array(hashes, dtype='uint64')
    corr = x @ x.T
    distance = np.bitwise_count(h[:, None] ^ h[None, :])
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    pairs=[]
    for i,j in zip(*np.where(np.triu((corr >= .985) & (distance <= 4), 1))):
        parent[find(int(j))] = find(int(i))
        pairs.append(dict(i=int(i),j=int(j),corr=float(corr[i,j]),dhash=int(distance[i,j])))
    for _, g in df.groupby('pixel_sha256'):
        ids=g.index.tolist()
        for j in ids[1:]: parent[find(j)]=find(ids[0])
    df['grupo'] = [find(i) for i in range(len(df))]
    pd.DataFrame(pairs, columns=['i','j','corr','dhash']).to_csv(RES/'pares_similares.csv',index=False)
    # Excluir grupos con conflicto de etiqueta; conservar solo una copia exacta.
    conflicts = df.groupby('grupo').label.nunique()
    bad = set(conflicts[conflicts>1].index)
    df['excluir'] = df.grupo.isin(bad) | df.duplicated('pixel_sha256')
    df[df.excluir].to_csv(RES/'exclusiones.csv',index=False)
    clean=df[~df.excluir].copy()
    clean['fold']=-1
    splitter=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=42)
    for k,(_,ids) in enumerate(splitter.split(clean,clean.label,clean.grupo)):
        clean.loc[clean.index[ids],'fold']=k
    clean['particion']=np.where(clean.fold<6,'train',np.where(clean.fold<8,'val','test'))
    assert clean.groupby('grupo').particion.nunique().max()==1
    assert clean.groupby('pixel_sha256').particion.nunique().max()==1
    clean.to_csv(RES/'particiones.csv',index=False)
    frames=[clean]
    ext=[]
    for label,clase in enumerate(CLASES):
        for p in sorted((DATA/'cgi_externo'/'rps-test-set'/clase).glob('*.png')):
            ext.append(dict(path=str(p.relative_to(ROOT)),clase=clase,label=label,grupo=p.stem.split('-')[0],particion='externo'))
    external=pd.DataFrame(ext)
    external.to_csv(RES/'externo.csv',index=False)
    for part, sub in pd.concat([clean,external]).groupby('particion'):
        arr=[]
        for path in sub.path:
            im=cv2.imread(str(ROOT/path)); im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            # Letterbox: no deformar la mano ni cortar los dedos; relleno fijo gris.
            height,width=im.shape[:2]; scale=SIZE/max(height,width)
            im=cv2.resize(im,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
            out=np.full((SIZE,SIZE,3),127,dtype='uint8')
            top=(SIZE-im.shape[0])//2;left=(SIZE-im.shape[1])//2
            out[top:top+im.shape[0],left:left+im.shape[1]]=im
            arr.append(out)
        np.savez_compressed(DATA/f'{part}_128.npz',X=np.stack(arr),y=sub.label.to_numpy(),paths=sub.path.to_numpy(dtype=str),groups=sub.grupo.to_numpy(dtype=str))
    summary={'n_original':len(df),'n_limpio':len(clean),'excluidos':int(df.excluir.sum()),'grupos':int(clean.grupo.nunique()),
             'pares_similares':len(pairs),'umbral_corr':.985,'umbral_dhash':4,
             'particiones':pd.crosstab(clean.particion,clean.clase).to_dict(),
             'sha256_particiones':hashlib.sha256((RES/'particiones.csv').read_bytes()).hexdigest(),
             'limite':'No existen identificadores de persona o sesión. Agrupar semejanza visual reduce fugas, pero no demuestra independencia entre sujetos.'}
    (RES/'datos_resumen.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    fig,axs=plt.subplots(2,3,figsize=(9,5))
    for row,part in enumerate(['train','externo']):
        a=np.load(DATA/f'{part}_128.npz')
        for c in range(3):
            axs[row,c].imshow(a['X'][np.flatnonzero(a['y']==c)[0]])
            axs[row,c].set_title(f'{NOMBRES[c]} · {"Real" if row==0 else "CGI externo"}');axs[row,c].axis('off')
    fig.tight_layout();fig.savefig(FIG/'datos_dominios.png',dpi=170);plt.close(fig)
    if pairs:
        fig,axs=plt.subplots(min(4,len(pairs)),2,figsize=(6,2*min(4,len(pairs))),squeeze=False)
        for row,pair in enumerate(sorted(pairs,key=lambda p:-p['corr'])[:4]):
            for col,idx in enumerate([pair['i'],pair['j']]):
                im=cv2.cvtColor(cv2.imread(str(ROOT/df.loc[idx,'path'])),cv2.COLOR_BGR2RGB)
                axs[row,col].imshow(im);axs[row,col].axis('off');axs[row,col].set_title(f'{df.loc[idx,"clase"]} · grupo {df.loc[idx,"grupo"]}')
        fig.tight_layout();fig.savefig(FIG/'grupos_visuales.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return clean, external

if __name__=='__main__': preparar()
