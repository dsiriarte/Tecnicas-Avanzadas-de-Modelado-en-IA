# Actividad 3. Transferencia y ajuste fino para reconocer gestos

David Santiago Iriarte Zamora y Jose Adrian Trochez Cuellar · Grupo 12

[**Abrir notebook ejecutado**](A3_transferencia_gestos_rps.ipynb) · [**Abrir informe PDF (5 páginas)**](Reporte_A3_Transferencia_Gestos_Grupo12_Iriarte_Trochez.pdf)

## Qué contiene esta carpeta

Usamos un caso de clasificación de papel, piedra y tijera con fuentes públicas. Cada archivo necesario para ejecutar esta actividad está dentro de esta carpeta. El notebook conserva sus resultados para revisarlo en GitHub; los módulos completos están en `src/`.

| Archivo o carpeta | Contenido |
|---|---|
| `A3_transferencia_gestos_rps.ipynb` | Desarrollo, código, resultados e interpretación |
| `Reporte_A3_Transferencia_Gestos_Grupo12_Iriarte_Trochez.pdf` | Informe técnico |
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
python -m pip install -r 'Unidad 3/Actividad 3/requirements_u3.txt'
cd 'Unidad 3/Actividad 3'
python -m notebook
```

En Windows, activar con `.venv\Scripts\activate`. El notebook encuentra sus módulos desde la carpeta de la actividad. Por defecto reutiliza los modelos y mediciones incluidos; no inicia un entrenamiento ni abre la ventana de video. La instalación de dependencias necesita internet, pero los datos procesados están incluidos.

El entorno original fue Python 3.12.11 en macOS arm64 y CPU M4 Pro. Las versiones están fijadas en `requirements_u3.txt`. El rendimiento puede variar en otros equipos. No instalar `opencv-python-headless` junto con `opencv-python` en este entorno de escritorio.

## Método y resultados

Preparamos 2.188 fotografías y las dividimos en 1.314 de entrenamiento, 437 de validación y 437 de prueba, agrupando imágenes casi duplicadas. Adaptamos MobileNetV2 con ImageNet y comparamos su base congelada con ajuste fino parcial en tres semillas. Guardamos los seis modelos en `modelos/`.

Ambas estrategias alcanzaron 437/437 aciertos en prueba real. La exactitud media sobre 372 imágenes CGI externas fue 62,72 % con base congelada y 57,80 % con ajuste fino. El resultado muestra límites ante cambios de fondo, orientación y textura; no afirmamos una mejora en prueba por ajuste fino.

Para volver a entrenar desde los datos incluidos:

```sh
python src/entrenar_a3.py --force
python src/analizar_a3.py
```

Para reconstruir los datos desde las fuentes públicas:

```sh
python src/descargar_datos.py
python src/preparar_datos.py
python src/preparar_pesos.py
```

En el notebook, `REENTRENAR=True` repite el entrenamiento y `RECONSTRUIR_DATOS=True` reconstruye los datos. Los valores predeterminados son False.

## Fuentes y licencias

Las fotografías y sus tensores derivados mantienen la atribución y licencia CC BY-SA 4.0 de DrGFreeman; las imágenes CGI de Laurence Moroney, CC BY 2.0. La demostración de video conserva la licencia MIT de su repositorio. Consultar [fuentes y condiciones completas](recursos_y_licencias.md). Las cifras del informe proceden de ejecuciones locales; no se modificaron al organizar este repositorio.
