"""Descarga fuentes públicas y conserva sus huellas; no requiere credenciales."""
from pathlib import Path
import hashlib
import json
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datos'
SOURCES = {
    'rps_real.zip': {
        'url': 'https://www.dropbox.com/s/sr1iui3zfdk0o53/rps-cv-image.zip?dl=1',
        'autor': 'DrGFreeman', 'licencia': 'CC BY-SA 4.0',
        'fuente': 'https://github.com/DrGFreeman/rps-cv-data-science',
        'extraer': 'real'},
    'rps_cgi_test.zip': {
        'url': 'https://storage.googleapis.com/learning-datasets/rps-test-set.zip',
        'autor': 'Laurence Moroney', 'licencia': 'CC BY 2.0',
        'fuente': 'https://laurencemoroney.com/datasets.html',
        'extraer': 'cgi_externo'},
    'rps_demo.gif': {
        'url': 'https://raw.githubusercontent.com/DrGFreeman/rps-cv/15ca5d41b97b9f69b6f90cd52d3664bd947070a6/img/doc/rps.gif',
        'autor': 'Julien de la Bruère-Terreault (DrGFreeman)', 'licencia': 'MIT (documentación del repositorio)',
        'fuente': 'https://github.com/DrGFreeman/rps-cv'},
    'LICENSE_demo.txt': {
        'url': 'https://raw.githubusercontent.com/DrGFreeman/rps-cv/15ca5d41b97b9f69b6f90cd52d3664bd947070a6/LICENSE',
        'autor': 'Julien de la Bruère-Terreault', 'licencia': 'MIT',
        'fuente': 'https://github.com/DrGFreeman/rps-cv'},
}

def descargar():
    DATA.mkdir(parents=True, exist_ok=True)
    registro = {}
    for name, spec in SOURCES.items():
        target = DATA / name
        if not target.exists():
            print('Descargando', name, flush=True)
            with requests.get(spec['url'], stream=True, timeout=(20, 120),
                              headers={'User-Agent': 'AcademicRPSStudy/1.0'}) as response:
                response.raise_for_status()
                temp = target.with_suffix(target.suffix + '.part')
                with temp.open('wb') as f:
                    for chunk in response.iter_content(1024 * 1024):
                        f.write(chunk)
                temp.replace(target)
        digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        registro[name] = {**spec, 'sha256': digest, 'bytes': target.stat().st_size}
        if 'extraer' in spec:
            dest = DATA / spec['extraer']
            if not dest.exists():
                dest.mkdir()
                with zipfile.ZipFile(target) as z:
                    for member in z.infolist():
                        out = (dest / member.filename).resolve()
                        if not out.is_relative_to(dest.resolve()):
                            raise ValueError('Ruta insegura en ZIP')
                    z.extractall(dest)
        print(name, target.stat().st_size, digest, flush=True)
        (DATA / 'fuentes.json').write_text(json.dumps(registro, indent=2, ensure_ascii=False))
    return registro

if __name__ == '__main__':
    descargar()
