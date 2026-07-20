# Scoring de crédito para pymes — SBA National Loans

Modelo de originación de crédito (predicción de impago en el momento de la concesión) y su marco de seguimiento en producción, construidos sobre el dataset público de préstamos de la U.S. Small Business Administration.

## Contexto

El dataset recoge ~899.000 préstamos concedidos por la SBA entre 1962 y 2014, con el estado final de cada uno (pagado en su totalidad o impago/*charge-off*). Es un dataset habitual en ejercicios de scoring, pero este proyecto no se queda en el EDA + modelo típico: incluye un ciclo completo de **desarrollo y seguimiento** — auditoría de calidad de datos, dos modelos comparados con criterios de banca (interpretabilidad, auditabilidad), y un marco de monitorización (PSI, estabilidad del Gini, calibración por cohorte) tal como se plantearía en una unidad de riesgos real, incluyendo sus limitaciones (censura, heterogeneidad temporal) en vez de ocultarlas.

## Arquitectura del proyecto

| Fase | Qué hace | Notebook / módulo |
|---|---|---|
| 1. Calidad de datos y EDA | Audita y corrige cada problema de calidad del CSV crudo (fechas, columnas monetarias como texto, categorías inconsistentes), define el target y delimita las variables de *leakage* | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) · [`src/data_cleaning.py`](src/data_cleaning.py) |
| 2. Modelo de originación | Entrena y compara un scorecard (WoE + regresión logística) y un challenger XGBoost, con split temporal y análisis de punto de corte por coste de negocio | [`notebooks/02_modelo.ipynb`](notebooks/02_modelo.ipynb) · [`src/modeling.py`](src/modeling.py) |
| 3. Seguimiento (monitorización) | Evalúa ambos modelos ya entrenados sobre cohortes nuevas (incluidas las excluidas del desarrollo por censura), midiendo PSI, Gini y calibración en el tiempo | [`notebooks/03_seguimiento.ipynb`](notebooks/03_seguimiento.ipynb) · [`src/monitoring.py`](src/monitoring.py) · [`reports/informe_seguimiento.md`](reports/informe_seguimiento.md) |

## Resultados clave

| Métrica | Scorecard (logit + WoE) | XGBoost (challenger) |
|---|---|---|
| Gini train | 0,795 | 0,966 |
| Gini test (2008-2011) | 0,640 | 0,930 |
| KS train | 0,672 | 0,882 |
| KS test | 0,491 | 0,810 |

XGBoost discrimina mejor en cualquier corte, pero la diferencia relevante para producción es de **estabilidad**: en el seguimiento posterior (Fase 3), el scorecard pierde entre 10 y 25 puntos porcentuales de Gini respecto a su nivel de entrenamiento en las cohortes 2008-2011 (ya maduras), frente a una caída de 2 a 8 puntos en XGBoost. El modelo más simple es también el que peor envejece.

## Hallazgos que van más allá de ejecutar el pipeline

- **La censura por la derecha determinó el propio diseño del split.** La Fase 1 detectó que las cohortes 2012-2014 tienen tasas de impago artificialmente bajas (aún no ha dado tiempo a hacer *default*) y las 1966-1988 tienen sesgo de muestra pequeña. El modelo se entrenó y evaluó solo con 1989-2011 (train 1989-2007, test 2008-2011); 2012-2014 se reservaron para el seguimiento, con la censura señalada explícitamente en cada métrica que las usa.
- **Se descartó una variable con IV de 0,56 por ser una fuga disfrazada, no señal de riesgo.** `urban_rural_label` parecía predictiva, pero resultó ser casi un proxy perfecto de "préstamo anterior a 1999" (la SBA no capturaba ese campo de forma sistemática antes de esa fecha), coincidiendo con el periodo de menor morosidad del dataset. Se excluyó del modelo con la investigación documentada en el notebook.
- **El PSI contra todo el conjunto de train no es una señal accionable por sí sola.** Al abarcar 19 años y varios ciclos económicos completos, el PSI ya es alto (hasta 0,5+) incluso comparando un año de train contra el propio agregado de entrenamiento. El cuadro de mando de seguimiento se restringió a partir del despliegue (2008 en adelante) para no confundir heterogeneidad histórica conocida con una alerta real.
- **XGBoost sobreestima la probabilidad de impago de forma sistemática, incluso en train** (entre 5 y 9 puntos porcentuales por encima de la tasa observada), efecto colateral de `scale_pos_weight` usado para compensar el desbalanceo de clases. Mejora el *ranking* (Gini) pero invalida sus probabilidades como estimación directa de pérdida esperada sin recalibrar.

## Recomendación final

Scorecard como modelo campeón en producción, por su interpretabilidad y auditabilidad ante una segunda línea de validación; XGBoost como *challenger* permanente para cuantificar cuánto poder discriminante se sacrifica por esa interpretabilidad. Dada su degradación sostenida desde 2009, se recomienda escalar la recalibración del scorecard antes del próximo ciclo anual. Las cohortes 2012-2014 quedan en seguimiento pasivo hasta que maduren lo suficiente para una lectura fiable. Detalle completo en [`reports/informe_seguimiento.md`](reports/informe_seguimiento.md).

## Cómo reproducirlo

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name=sme-credit-scoring --display-name "Python (sme-credit-scoring)"
```

1. Descarga `SBAnational.csv` (ver fuente de datos) y colócalo en `data/raw/SBAnational.csv`.
2. Ejecuta los notebooks en orden, seleccionando el kernel `sme-credit-scoring`: `01_eda.ipynb` → `02_modelo.ipynb` → `03_seguimiento.ipynb`. Cada uno puede ejecutarse de arriba abajo sin pasos manuales intermedios; los artefactos de una fase (dataset limpio, modelos serializados) alimentan la siguiente.

## Estructura de carpetas

```
data/raw/           CSV crudo (no versionado salvo vía Git LFS)
data/processed/      Dataset limpio (sba_clean.parquet)
notebooks/           01_eda · 02_modelo · 03_seguimiento
src/                 data_cleaning.py · modeling.py · monitoring.py
models/              Modelos serializados (scorecard, WoE bins, XGBoost)
figures/             Gráficos exportados por los tres notebooks
reports/             Informe de seguimiento para comité
requirements.txt     Entorno fijado (pandas, scikit-learn, optbinning, xgboost, shap...)
```

## Fuente de datos

Dataset ["Should This Loan be Approved or Denied?"](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied) (Kaggle), basado en los datos públicos de la SBA y en la guía de trabajo de Li, X., Mickel, A., & Taylor, S. (2018). *Should This Loan Be Approved or Denied?: A Large Dataset with Class Assignment Guidelines*. Journal of Statistics Education, 26(1).
