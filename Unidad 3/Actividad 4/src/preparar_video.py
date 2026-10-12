"""Conversión documentada del GIF público a video decodificable por OpenCV."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def preparar():
    source=ROOT/'datos/rps_demo.gif';target=ROOT/'datos/rps_demo.avi'
    if target.exists():return target
    im=Image.open(source);durations=[]
    # Cadencia fija de 70 ms: conserva todos los cuadros, alarga levemente el GIF.
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'MJPG'),1000/70,im.size)
    if not writer.isOpened():raise RuntimeError('No se pudo crear el AVI MJPG')
    for k in range(im.n_frames):
        im.seek(k);durations.append(im.info.get('duration',70))
        writer.write(cv2.cvtColor(np.array(im.convert('RGB')),cv2.COLOR_RGB2BGR))
    writer.release()
    meta={'source':source.name,'frames':im.n_frames,'width':im.width,'height':im.height,
          'duration_ms_values':sorted(set(durations)),'duration_s':sum(durations)/1000,
          'fps_contenedor':1000/70,'duracion_avi_s':im.n_frames*.07,'codec':'MJPG',
          'nota':'Todos los cuadros originales; cadencia normalizada a 70 ms. El benchmark no limita la lectura a esta cadencia.'}
    (ROOT/'datos/video_metadata.json').write_text(json.dumps(meta,indent=2))
    return target

if __name__=='__main__':print(preparar())
