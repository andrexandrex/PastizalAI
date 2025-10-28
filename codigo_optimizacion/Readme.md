# Código de optimización (codigo_optimizacion)

Resumen
-------
Esta carpeta debe agrupar scripts y notebooks destinados a optimizar la asignación de recuperación de pixeles, hiperparámetros o pipelines (ej.: GridSearch, Bayesian optimization, tuning de hiperparámetros, optimización de preprocesamiento).

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
  - pyomo

Ejecutar optimizaciones (ejemplo)
---------------------------------
1. Preparar entorno e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar script de optimización (ejemplo):
   ```bash
   pip install -r requirements.txt

  # 3) Prepare data:
  python prepare_data.py
  
  # 4) Optimize:
  python optimize_allocation.py
   ```
3. Guardar resultados como  budget_01_selection y budget_02_selection para los dos presupuestos: 23,000$ y 90,000$

Reproducibilidad
----------------
- Guarda el archivo de configuración con los espacios de búsqueda y el seed.
- Versiona los resultados y el set de datos usados (o registra su checksum).
- Limita el número de hilos si usas paralelización para tener consistencia:
  ```bash
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
Instalación de solver glpk

Resultados de optimización se pueden encontrar en opti_pastizalAI_final.ipynb que permite visualización y el cálculo de la estimación de beneficio económico del ppt (anual)
