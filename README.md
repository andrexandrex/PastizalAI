# 🛰️ PastizalAI: La brújula inteligente para revivir Picotani 🧭

## Descripción 🎯

Proyecto para el **análisis, diagnóstico y predicción** sobre pastizales altoandinos (p. ej., carbono, biomasa y cobertura vegetal).

Este repositorio centraliza el desarrollo de modelos de machine learning y la optimización de *pipelines* robustos, sentando las bases para un potencial **Sistema de Alerta Temprana (SAT)** de degradación de suelos en la zona de Picotani.

---

## Estructura del Repositorio 📂

El proyecto se organiza en los siguientes módulos principales:

* **`/Cod_Diagnostico/`** 🔬
    * Notebooks para la limpieza de datos, análisis exploratorio (EDA) y desarrollo de los modelos base de predicción (Ej.: `Prediccion_carbono.ipynb`, `Prediccion_cobertura.ipynb`).

* **`/codigo_optimizacion/`** 📈
    * Scripts y notebooks dedicados a la búsqueda y ajuste fino de hiperparámetros, experimentación y *tracking* de modelos (MLflow, etc.).

* **`/Cod_Sistema_Alerta_Temprana/`** ⚠️
    * Pipelines y notebooks orientados a la detección temprana de anomalías o cambios significativos en series temporales y observaciones espaciales.

* **`/Cod_Shiny/`** 🖥️
    * Código fuente del **Dashboard Interactivo** (desarrollado en Shiny) para la visualización de mapas, tendencias y resultados del análisis.
    
---

*Nota: Cada uno de los folderes (`/Cod_Diagnostico/`, `/codigo_optimizacion/`, etc.) contiene su propio `README.md` secundario, explicando en detalle la inicialización y los pasos para la reproducibilidad de esa sección.*