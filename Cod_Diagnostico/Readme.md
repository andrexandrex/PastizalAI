# Resultados de Evaluación de Modelos para Análisis Geoespacial

Este repositorio contiene los resultados de la evaluación de varios modelos de machine learning desarrollados para predecir variables ambientales clave. El objetivo final es generar mapas de predicción para el análisis de la degradación, la biomasa y las reservas de carbono.

---

## Productos Generados: Mapas de Predicción

Los modelos evaluados a continuación se utilizaron para generar los siguientes productos cartográficos:

1.  **Mapa de Carbono Stock:** Estimación de la distribución espacial del Carbono Orgánico del Suelo (COS).
2.  **Mapa de Biomasa (Dry_Weigth):** Predicción de la biomasa aérea (peso seco) en los pastizales.
3.  **Mapa de Erosividad:** Zonificación del riesgo potencial de erosión del suelo.
4.  **Mapa de Cobertura de Suelo:** Clasificación de los diferentes tipos de cobertura vegetal y suelo (bosque, pastizal, suelo desnudo, etc.).
5.  **Mapa de Políticas Potenciales:** Un mapa de síntesis que cruza los resultados anteriores para identificar zonas críticas de intervención, conservación o restauración.

A continuación, se presentan las métricas de rendimiento detalladas de los modelos utilizados para crear estos mapas.

---
---

## Modelos de Regresión (Carbono Stock, Erosividad, Dry_weigth)

Estas son las métricas de los modelos que predicen valores numéricos continuos (ej. toneladas de carbono, kg de biomasa).

### Tabla 1: Resultados (Carbono Stock / Erosividad)

| Modelo | MAE_Train | RMSE_Train | R2_Train | MAPE_Train | MAE_Test | RMSE_Test | R2_Test | MAPE_Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | 45.5706 | 68.6301 | 0.2876 | 46.3708 | 61.4119 | 96.9094 | 0.1405 | 51.6752 |
| **Decision Tree**| 38.5177 | 51.9730 | 0.5914 | 34.9618 | 54.3312 | 83.8683 | 0.3563 | 36.9141 |
| **Random Forest**| 40.6413 | 54.4810 | 0.5511 | 41.5757 | 55.3321 | 82.7808 | 0.3729 | 45.4249 |
| **Lasso** | 46.9922 | 61.6285 | 0.4255 | 49.8705 | 56.0269 | 87.8343 | 0.2939 | 50.6529 |
| **Ride** | 49.6615 | 64.7463 | 0.3660 | 57.3545 | 59.3587 | 91.9216 | 0.2267 | 59.6574 |
| **XGBoost** | 42.8573 | 57.1192 | 0.5065 | 50.8197 | 63.3903 | 92.8772 | 0.2105 | 59.8290 |

<br>

### Tabla 2: Resultados (Dry_weigth / Biomasa)

| Modelo | MAE_Train | RMSE_Train | R2_Train | MAPE_Train | MAE_Test | RMSE_Test | R2_Test | MAPE_Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | 44.9953 | 69.5182 | 0.4334 | 21.4435 | 57.2740 | 92.6909 | 0.1185 | 25.4589 |
| **Decision Tree**| 30.7456 | 42.9340 | 0.7839 | 15.6718 | 56.8591 | 77.0491 | 0.3909 | 36.4454 |
| **Random Forest**| 30.1296 | 41.1143 | 0.8018 | 15.6195 | 54.9898 | 72.3021 | 0.4636 | 31.4306 |
| **Lasso** | 48.8859 | 63.7034 | 0.5242 | 25.7945 | 56.3495 | 81.5658 | 0.3174 | 27.8367 |
| **Ride** | 49.7812 | 65.0676 | 0.5036 | 26.0910 | 56.3258 | 81.8473 | 0.3127 | 27.7901 |
| **XGBoost** | 10.8327 | 13.7275 | 0.9779 | 5.6343 | 59.9926 | 75.7634 | 0.4111 | 32.0554 |

---

### Explicación de Métricas de Regresión

* **`_Train` vs. `_Test`**: `_Train` (Entrenamiento) mide el rendimiento en los datos de entrenamiento. `_Test` (Prueba) mide el rendimiento en datos nuevos y es la métrica más importante para evaluar la generalización.
* **MAE (Error Absoluto Medio)**: El promedio de los errores de predicción. **Objetivo: Bajarlo.**
* **RMSE (Raíz del Error Cuadrático Medio)**: Similar al MAE, pero penaliza más los errores grandes. **Objetivo: Bajarlo.**
* **R2 (Coeficiente de Determinación)**: Proporción de la varianza explicada por el modelo. Va de 0 a 1. **Objetivo: Subirlo (cerca de 1).**
* **MAPE (Error Porcentual Absoluto Medio)**: El MAE expresado como un porcentaje. **Objetivo: Bajarlo.**

---
---

## Modelos de Clasificación (Predicción de Cobertura de Suelo)

Estos son los resultados de los modelos entrenados para predecir la cobertura de suelo (un problema de clasificación).

### Tabla 3: Tabla Comparativa de Modelos de Clasificación

| Modelo | Accuracy_Train | F1_Train (w) | Precision_Train (w) | Recall_Train (w) | Accuracy_Test | F1_Test (w) | Precision_Test (w) | Recall_Test (w) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 0.9441 | 0.9418 | 0.9476 | 0.9441 | 0.7765 | 0.7566 | 0.7408 | 0.7765 |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7647 | 0.7542 | 0.7501 | 0.7647 |
| **SVM (SVC)** | 0.7941 | 0.7701 | 0.7654 | 0.7941 | 0.7647 | 0.7406 | 0.7427 | 0.7647 |
| **Lasso (LogReg)**| 0.7824 | 0.7590 | 0.7496 | 0.7824 | 0.7647 | 0.7399 | 0.7311 | 0.7647 |
| **Decision Tree**| 0.8529 | 0.8333 | 0.8706 | 0.8529 | 0.7529 | 0.7317 | 0.7254 | 0.7529 |
| **Ridge (LogReg)** | 0.7882 | 0.7647 | 0.7541 | 0.7882 | 0.7529 | 0.7287 | 0.7314 | 0.7529 |

*Nota: La (w) probablemente se refiere a "weighted" (ponderado), un promedio que considera el desbalance entre clases.*

### Explicación de Métricas de Clasificación

* **Accuracy (Exactitud)**: El porcentaje de predicciones correctas. `(Predicciones Correctas / Total de Predicciones)`. Es útil, pero puede ser engañoso si las clases están desbalanceadas.
* **Precision (Precisión)**: De todas las veces que el modelo predijo una clase, ¿qué porcentaje acertó? Mide la calidad de la predicción.
* **Recall (Sensibilidad)**: De todos los valores reales de una clase, ¿qué porcentaje el modelo fue capaz de encontrar? Mide la cantidad de positivos que el modelo "capturó".
* **F1-Score**: La media armónica entre Precision y Recall. Es una métrica única que balancea ambas. Es una de las mejores métricas para evaluar el rendimiento general en clasificación.

**Objetivo para todas las métricas de clasificación:** **Cuanto más alto (cercano a 1.0 o 100%), mejor.**

**Análisis Rápido (Tabla 3):**
* **Random Forest** y **XGBoost** muestran un claro **sobreajuste (overfitting)**. Tienen un rendimiento perfecto en `_Train` pero su rendimiento cae significativamente en `_Test`.
* Los demás modelos (SVM, Lasso, Ridge) son más estables, con rendimientos similares en Train y Test, aunque su rendimiento general es ligeramente inferior al de XGBoost en Test.