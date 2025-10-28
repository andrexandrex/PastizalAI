# Código de optimización (codigo_optimizacion)

Resumen
-------
Esta carpeta debe agrupar scripts y notebooks destinados a optimizar modelos, hiperparámetros o pipelines (ej.: GridSearch, Bayesian optimization, tuning de hiperparámetros, optimización de preprocesamiento).

Contenido esperado
------------------
Sugerencias de archivos típicos que deberían estar o añadirse:
- notebooks/optimización_hyperparametros.ipynb
- scripts/optimize.py
- experiments/ (resultados, registros)
- configs/ (parámetros de experimentos)

Requisitos / Entorno
--------------------
- Utilizar el mismo entorno que el resto del proyecto (ver Cod_Diagnostico).
- Dependencias típicas:
  - scikit-learn
  - optuna o hyperopt
  - joblib
  - pandas, numpy

Ejecutar optimizaciones (ejemplo)
---------------------------------
1. Preparar entorno e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar script de optimización (ejemplo):
   ```bash
   python scripts/optimize.py --config configs/opt_config.yaml
   ```
3. Guardar resultados en `experiments/<fecha>/` con métricas y configuraciones.

Reproducibilidad
----------------
- Guarda el archivo de configuración con los espacios de búsqueda y el seed.
- Versiona los resultados y el set de datos usados (o registra su checksum).
- Limita el número de hilos si usas paralelización para tener consistencia:
  ```bash
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  ```

Buenas prácticas
----------------
- Incluye logs con timestamps.
- Usa experiment tracking (MLflow, Weights & Biases o un simple CSV/JSON de resultados).
- Documenta la métrica objetivo y el criterio de parada (evaluación en validación cruzada, holdout).
