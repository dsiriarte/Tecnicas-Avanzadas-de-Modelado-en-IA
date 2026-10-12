# Actividad 4. Optimización de un pipeline de visión con hilos

David Santiago Iriarte Zamora y Jose Adrian Trochez Cuellar · Grupo 12

[**Abrir notebook ejecutado**](A4_pipeline_concurrente_rps.ipynb) · [**Abrir informe PDF (5 páginas)**](Reporte_A4_Eficiencia_Gestos_Grupo12_Iriarte_Trochez.pdf)

## Qué contiene esta carpeta

Usamos un caso de clasificación de papel, piedra y tijera con fuentes públicas. Cada archivo necesario para ejecutar esta actividad está dentro de esta carpeta. El notebook conserva sus resultados para revisarlo en GitHub; los módulos completos están en `src/`.

| Archivo o carpeta | Contenido |
|---|---|
| `A4_pipeline_concurrente_rps.ipynb` | Desarrollo, código, resultados e interpretación |
| `Reporte_A4_Eficiencia_Gestos_Grupo12_Iriarte_Trochez.pdf` | Informe técnico |
| `src/` | Implementación completa |
| `datos/` | Datos preparados y fuentes |
| `resultados/` | Métricas, configuraciones y evidencia |
| `figuras/` | Visualizaciones incluidas en el informe |
| `recursos_y_licencias.md` | Atribuciones y condiciones de uso |

## Ejecución

Desde la raíz del repositorio:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r 'Unidad 3/Actividad 4/requirements_u3.txt'
cd 'Unidad 3/Actividad 4'
python -m notebook
```

En Windows, activar con `.venv\Scripts\activate`. El notebook encuentra sus módulos desde la carpeta de la actividad. Por defecto reutiliza los modelos y mediciones incluidos; no inicia un entrenamiento ni abre la ventana de video. La instalación de dependencias necesita internet, pero los datos procesados están incluidos.

El entorno original fue Python 3.12.11 en macOS arm64 y CPU M4 Pro. Las versiones están fijadas en `requirements_u3.txt`. El rendimiento puede variar en otros equipos. No instalar `opencv-python-headless` junto con `opencv-python` en este entorno de escritorio.

## Método y resultados

El modelo seleccionado en A3 y su configuración están copiados en `modelo/`. Esta actividad se ejecuta sin acceder a la carpeta de A3. Conservamos el video público, los CSV por cuadro y dos conjuntos etiquetados para comprobar equivalencia predictiva.

Comparamos ejecución secuencial y con hilos, tanto en modo eager como en grafo. La referencia logró 22,25 ± 0,30 FPS; la versión con grafo, hilos y colas de capacidad 1, 80,79 ± 0,23 FPS. Son cuadros presentados por el ciclo visual OpenCV sobre un archivo acelerado, no una medida del refresco físico del monitor ni de captura de nuevas imágenes a esa frecuencia. No hubo cambios de clase por la optimización.

Desde esta carpeta, con el entorno activado:

```sh
# Comparación factorial con ventana.
python src/pipeline_a4.py --display --reps 3 --tag nueva
# Configuración final.
python src/pipeline_a4.py --display --reps 3 --modes hilos_grafo --queue-size 1 --tag nueva_cola1
# Entrada controlada a 60 FPS durante aproximadamente un minuto.
python src/pipeline_a4.py --display --reps 1 --modes hilos_grafo --queue-size 1 --loops 4 --pace --pace-fps 60 --tag nueva_60
# Comprobar calidad y manejo de errores.
python src/verificar_a4.py
python src/verificar_concurrencia.py
# Rehacer figuras desde las mediciones originales.
python src/analizar_a4.py
```

La ventana requiere una sesión de escritorio. La tecla q cancela la reproducción. Sin `--display`, el resultado mide procesamiento sin presentación. Los tags nuevos conservan las mediciones originales. No ejecutar trabajos intensivos en paralelo con el benchmark.

`resultados/pipeline_version_medida.txt` conserva la versión de referencia del benchmark con la estructura original de carpetas; es una copia para trazabilidad, no el punto de entrada. Ejecutar `src/pipeline_a4.py`, que adapta las rutas a esta carpeta y refuerza manejo de errores y cancelación. Las pruebas simuladas de concurrencia no se usan para calcular FPS.

## Fuentes y licencias

Las fotografías y sus tensores derivados mantienen la atribución y licencia CC BY-SA 4.0 de DrGFreeman; las imágenes CGI de Laurence Moroney, CC BY 2.0. La demostración de video conserva la licencia MIT de su repositorio. Consultar [fuentes y condiciones completas](recursos_y_licencias.md). Las cifras del informe proceden de ejecuciones locales; no se modificaron al organizar este repositorio.
