# Unidad 4 · Pronóstico a 10 minutos de la actividad de Internet en Milán

David Santiago Iriarte Zamora y Jose Adrian Trochez Cuellar · Grupo 12

Pronosticamos el siguiente intervalo de 10 minutos de la actividad de Internet agregada de Milán (Telecom Italia Big Data Challenge, 4 al 10 de noviembre de 2013) con redes recurrentes LSTM y GRU, y las comparamos con líneas base ingenuas sobre la misma partición cronológica.

## Documentos

- **Actividad 2.** `Actividad 2/Reporte_A2_Pipeline_Avanzado_Grupo12_Iriarte_Trochez.pdf`: reporte técnico del pipeline, la optimización de hiperparámetros, las curvas de aprendizaje y la comparación con las líneas base.
- **Actividad 3.** `Actividad 3/Informe_Final_A3_Grupo12_Iriarte_Trochez.pdf`: informe final, con la ficha del modelo y la ficha de los datos en los anexos. `Actividad 3/Presentacion_A3_Grupo12_Iriarte_Trochez.pptx`: presentación de la sustentación.

## Estructura

```
Unidad 4/
  README.md                     este archivo
  requirements_u4.txt           versiones de la ejecución reportada
  datos/
    milan_internet_aggregated.csv   serie agregada (1.008 intervalos de 10 min)
    milan_data_audit.json           auditoría de los 7 archivos diarios originales
  notebooks/
    U4_pronostico_internet_milan.ipynb   pipeline completo, ejecutado
  modelos/
    modelo_final_*_semilla{42,7,2026}.keras   pesos entrenados de la configuración final
    metadatos_modelo_final.json               configuración, escaladores, métricas y versiones
  salidas/                      figuras, experimentos.csv, comparación y contrastes en prueba
  mlruns/                       registro MLflow de todas las corridas
  _build/                       generadores del notebook y de la presentación, lecturas y estilos de los PDF
  Actividad 2/                  reporte técnico (PDF y HTML fuente)
  Actividad 3/                  informe final (PDF y HTML fuente) y presentación (PowerPoint)
```

## Reproducir

Con Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements_u4.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/U4_pronostico_internet_milan.ipynb
```

- El notebook encuentra solo la carpeta `datos/` (no hay rutas absolutas) y verifica el SHA-256 del CSV. En Colab, si el CSV no está, lo descarga de este repositorio.
- Corre de principio a fin en CPU en unos 6 minutos (Mac M4 Pro): 147 entrenamientos (3 rondas, 3 semillas cada configuración) más el reentrenamiento y guardado de los pesos finales.
- Semillas 42, 7 y 2026 con `enable_op_determinism`. La última celda recarga los `.keras` guardados y comprueba que reproducen el pronóstico de prueba reportado.
- Para ver el registro de experimentos: `mlflow ui --backend-store-uri ./mlruns` (requiere el paquete `mlflow` completo; `mlflow-skinny` solo escribe el registro). El mismo registro está en `salidas/experimentos.csv`.
- Para regenerar el notebook desde el código fuente: `python _build/build_u4_nb.py`.

## Resultado principal

En prueba (domingo, 152 ventanas), la GRU que predice el cambio respecto al último valor obtiene MAE 5.835, frente a 9.230 de la persistencia: −36,8 %, IC 95 % [−5.503; −902], Diebold-Mariano p < 0,001. Frente a Ridge con las mismas entradas, −12,7 % (p < 0,001). Frente a la deriva, −12,8 %, pero no significativo (p = 0,06). Detalle en los PDF y en `salidas/`.

## Datos y licencia

Barlacchi et al. (2015), *A multi-source dataset of urban life in the city of Milan and the Province of Trentino*, Scientific Data 2, 150055, https://doi.org/10.1038/sdata.2015.55. Los datos del Telecom Italia Big Data Challenge se publican bajo Open Database License (ODbL), https://theodi.fbk.eu/openbigdata . El CSV de esta carpeta es una agregación derivada: para cada marca de 10 minutos sumamos la columna `internet` de las 10.000 celdas y todos los códigos de país (`internet_sum`) y contamos los registros con `internet` no nulo (`observations`). Los valores ausentes no se imputan como cero. `internet_sum` es un índice de actividad, no bytes ni usuarios.
