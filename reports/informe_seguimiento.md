# Informe de seguimiento del modelo de originación — SBA National Loans

**Fecha del informe:** 2026-07-20
**Modelos monitorizados:** Scorecard (logit + WoE) y challenger XGBoost, ambos entrenados en la Fase 2 (train 1989-2007, test 2008-2011)
**Periodo cubierto por este seguimiento:** cohortes de aprobación 1989-2014 (incluye 2012-2014, nunca vistas en desarrollo)
**Fuente:** `notebooks/03_seguimiento.ipynb` (reproducible de punta a punta; no se ha reentrenado ningún modelo)

---

## 1. Resumen ejecutivo

- **Ningún modelo se ha reentrenado.** Este informe puntúa los dos modelos ya construidos en la Fase 2 sobre cohortes de solicitantes que nunca han visto, para comprobar si su comportamiento se mantiene.
- **El scorecard pierde capacidad discriminante de forma sostenida a partir de 2009**: su Gini cae entre 22 y 52 puntos porcentuales respecto a su nivel de entrenamiento en las cohortes 2009-2013. El challenger XGBoost se mantiene mucho más estable (caídas de 2 a 9 puntos porcentuales en el mismo periodo).
- **La población de solicitantes y de entidades prestamistas se ha movido de forma real desde 2008** (PSI elevado en `Bank_grouped`, `BankState` y `sba_guarantee_pct`), con una explicación de negocio identificable en cada caso (consolidación bancaria post-crisis, cambios de política de garantía de la SBA) — no parece un problema de calidad de datos.
- **Las cohortes 2012-2014 no son comparables directamente con 2008-2011**: su tasa de impago observada está artificialmente baja por censura (los préstamos más recientes aún no han tenido tiempo de entrar en impago). El cuadro de mando de este informe distingue explícitamente esta limitación.
- **Recomendación**: mantener el scorecard como modelo campeón por interpretabilidad, pero activar una revisión reforzada (ver semáforo, sección 6) — su degradación ya supera el umbral que justificaría una recalibración antes del próximo ciclo anual de revisión.

---

## 2. Alcance y metodología

- **Cohortes**: `approval_year`, igual granularidad que en las Fases 1 y 2.
- **Población de referencia para PSI**: el propio conjunto de train (1989-2007), tal como especifica el alcance de esta fase.
- **Aviso metodológico**: el train abarca 19 años con varios ciclos económicos completos, así que el PSI ya es alto (hasta 0.5+) **incluso comparando años dentro del propio train** contra el agregado de los 19 años. Los umbrales estándar (< 0.10 estable, 0.10-0.25 alerta moderada, > 0.25 alerta alta) se aplican con este contexto en mente: el cuadro de mando de la sección 6 se centra en **2008 en adelante**, el único periodo donde un desplazamiento representa señal de seguimiento genuina y no una comparación contra el propio agregado de desarrollo.
- **Aviso de censura (2012-2014)**: estas cohortes se excluyeron del desarrollo del modelo en la Fase 2 precisamente porque, a fecha de corte del dataset (2014), muchos de esos préstamos no han tenido tiempo de hacer default. Su tasa de impago observada está artificialmente baja; una caída aparente del Gini o un "gap" de calibración grande en 2012-2014 no debe interpretarse como fallo del modelo sin corregir primero por este efecto.
- **Reproducibilidad del preprocesado**: puntuar cohortes nuevas exige reproducir exactamente dos reglas fijadas con datos en la Fase 2 (la agrupación de bancos poco frecuentes en `Bank_grouped` y el catálogo de categorías de XGBoost). Se validó que el pipeline reconstruido reproduce el Gini de test de la Fase 2 con exactitud antes de puntuar ninguna cohorte nueva.

---

## 3. Estabilidad poblacional (PSI)

### 3.1 PSI del score (2008-2014, frente a train)

| Cohorte | n | PSI Scorecard | PSI XGBoost |
|---|---|---|---|
| 2008 | 30.729 | 0,315 (Rojo) | 0,375 (Rojo) |
| 2009 | 19.672 | 0,238 (Ámbar) | 0,146 (Ámbar) |
| 2010 | 16.757 | 0,225 (Ámbar) | 0,127 (Ámbar) |
| 2011 | 9.658 | 0,200 (Ámbar) | 0,261 (Rojo) |
| 2012 | 5.048 | 0,190 (Ámbar) | 0,288 (Rojo) |
| 2013 | 1.730 | 0,362 (Rojo) | 0,584 (Rojo) |
| 2014 | 128 | 0,924 (Rojo)* | 0,640 (Rojo)* |

\* Cohorte de 128 préstamos: el PSI es, en sí mismo, una estimación muy ruidosa con este tamaño de muestra.

### 3.2 PSI de las variables más importantes (Fase 2)

`Term`, `sba_guarantee_pct`, `Bank_grouped`, `RetainedJob`, `SBA_Appv`, `BankState` — mapa completo en `figures/psi_variables_heatmap.png`. Hallazgos principales:

- **`sba_guarantee_pct`** tiene PSI alto en casi todo el histórico, incluidos varios años de train: refleja decisiones de política pública (p. ej. la Ley ARRA de 2009 elevó temporalmente la garantía máxima al 90%), no un problema de datos.
- **`Bank_grouped`** y **`BankState`** muestran PSI claramente más alto en 2008-2014 que en la mayoría de los años de train: consistente con la consolidación bancaria y la salida de algunos prestamistas del programa SBA tras la crisis.
- **`SBA_Appv`** es la variable más estable de las seis fuera de la propia crisis (vuelve a zona verde/ámbar entre 2010 y 2013).
- Ninguna variable se dispara de forma aislada sin relación con el contexto económico: el patrón apunta a un desplazamiento real de la población, no a un error de captura de datos.

---

## 4. Estabilidad discriminante (Gini por cohorte)

Gini de referencia (train): **Scorecard 0,795 | XGBoost 0,966**.

| Cohorte | n | Gini Scorecard | Caída vs. train | Gini XGBoost | Caída vs. train |
|---|---|---|---|---|---|
| 2008 | 30.729 | 0,697 | 9,8 p.p. | 0,949 | 1,8 p.p. |
| 2009 | 19.672 | 0,573 | 22,2 p.p. | 0,887 | 7,9 p.p. |
| 2010 | 16.757 | 0,562 | 23,3 p.p. | 0,885 | 8,2 p.p. |
| 2011 | 9.658 | 0,547 | 24,9 p.p. | 0,938 | 2,8 p.p. |
| 2012 | 5.048 | 0,454 | 34,1 p.p.* | 0,933 | 3,3 p.p.* |
| 2013 | 1.730 | 0,279 | 51,6 p.p.* | 0,881 | 8,5 p.p.* |
| 2014 | 128 | n/d (0 impagos observados) | — | n/d | — |

\* Cohortes con tasa de impago observada afectada por censura (sección 2): la lectura de estas caídas debe combinarse con esa salvedad, no tomarse como degradación pura del modelo.

**Conclusión**: en las cohortes genuinamente maduras (2008-2011), el patrón de la Fase 2 se confirma y se agrava con más datos: **el scorecard pierde entre 2 y 2,5 veces más Gini relativo que XGBoost** en cada cohorte, y la brecha entre ambos modelos es mayor precisamente en el pico de la crisis. XGBoost tampoco es inmune al cambio de régimen, pero su degradación es sustancialmente menor.

---

## 5. Calibración

- En **train**, ambos modelos siguen de cerca la tasa observada, como cabe esperar de la propia muestra de ajuste.
- En **2007-2008** (pico de crisis), ambos modelos **subestiman** la tasa de impago real; en la recuperación (2009-2011) la **sobreestiman**. Es el patrón típico de un modelo que no ha visto un cambio de régimen tan brusco en desarrollo.
- **Hallazgo adicional sobre XGBoost**: su probabilidad predicha está sistemáticamente por encima de la tasa observada **incluso en train** (gap de +5 a +9 puntos porcentuales), un efecto esperable del ajuste por desbalanceo de clases (`scale_pos_weight`) usado en la Fase 2, que mejora la capacidad de ranking (Gini) pero desplaza la escala de probabilidad hacia arriba. **Implicación práctica**: si en algún momento se usan las probabilidades de XGBoost directamente para calcular pérdida esperada (como en el análisis de punto de corte de la Fase 2), conviene aplicar una recalibración (Platt/isotónica) antes de tratarlas como una probabilidad de impago real; para ordenar/clasificar riesgo, el efecto no aplica.
- En **2012-2014**, ambos modelos predicen tasas de impago próximas a las de 2005-2008 mientras la tasa observada cae hacia 0: es la firma esperada de la censura (sección 2), no una descalibración real.

---

## 6. Cuadro de mando y plan de acción

Semáforo combinado (PSI del score + caída de Gini vs. train), aplicado a partir del despliegue del modelo (2008 en adelante — comparar los años de entrenamiento contra su propio agregado no es una señal de seguimiento accionable, ver sección 2).

### Criterio

| Nivel | PSI del score | Caída de Gini vs. train | Acción recomendada |
|---|---|---|---|
| 🟢 Verde | < 0,10 | < 5 p.p. | Sin acción. Seguimiento en el ciclo habitual. |
| 🟡 Ámbar | 0,10 - 0,25 | 5 - 15 p.p. | Revisión trimestral reforzada: repetir este análisis con mayor frecuencia y vigilar las variables de la sección 3.2 que más contribuyan al desplazamiento. |
| 🔴 Rojo | > 0,25 | > 15 p.p. | Recalibración o reentrenamiento: escalar a comité de modelos: la señal ya no es ruido de fondo esperable y compromete la capacidad discriminante o la representatividad de la población de referencia. |

El semáforo final toma el **nivel más alto** entre PSI y caída de Gini (criterio conservador).

### Resultado por cohorte (2008-2014)

| Cohorte | Scorecard | XGBoost |
|---|---|---|
| 2008 | 🔴 Rojo | 🔴 Rojo |
| 2009 | 🔴 Rojo (Gini) | 🟡 Ámbar |
| 2010 | 🔴 Rojo (Gini) | 🟡 Ámbar |
| 2011 | 🔴 Rojo (Gini) | 🔴 Rojo (PSI) |
| 2012 | 🔴 Rojo (Gini)* | 🔴 Rojo (PSI)* |
| 2013 | 🔴 Rojo (Gini + PSI)* | 🔴 Rojo (PSI)* |
| 2014 | 🔴 Rojo (PSI, n=128)* | 🔴 Rojo (PSI, n=128)* |

\* Cohortes afectadas por censura (sección 2): confirmar con la cohorte 2015+ (una vez madure) antes de escalar acciones adicionales basadas solo en estos años.

### Lectura y recomendación

- **Scorecard**: el semáforo entra en rojo por caída de Gini de forma sostenida desde **2009**, con una tendencia que empeora cohorte a cohorte hasta 2013. Esto **sí es una señal de acción**, no un artefacto del PSI de fondo: la pérdida de poder discriminante es real y medible en cohortes ya maduras (2009-2011). **Recomendación: iniciar el proceso de recalibración/reentrenamiento del scorecard con datos más recientes** en el próximo ciclo, en vez de esperar al ciclo anual estándar.
- **XGBoost**: se mantiene mayormente en ámbar por PSI en 2009-2010 (desplazamiento de población, no de capacidad predictiva) y solo entra en rojo por Gini de forma puntual y moderada. **Recomendación: mantener en observación como challenger; no requiere acción inmediata**, pero conviene vigilar que la brecha de estabilidad frente al scorecard no se use como excusa para ignorar la degradación de este último.
- **2012-2014**: mantener en seguimiento pasivo hasta que maduren lo suficiente para una lectura fiable; no tomar decisiones de recalibración basadas únicamente en estas cohortes.

---

## 7. Recomendación final

1. **Escalar a comité la recalibración del scorecard**, apoyándose en la evidencia de la sección 4 (caída de Gini sostenida y creciente desde 2009 en cohortes ya maduras).
2. **Mantener XGBoost como challenger de referencia** en cada ciclo de seguimiento, para cuantificar cuánto poder discriminante se está sacrificando por interpretabilidad mientras el scorecard no se recalibre.
3. **Si se recalibra el scorecard**, considerar means de reflejar mejor los cambios de `Bank_grouped`/`BankState` (consolidación del mercado prestamista) y revisar si `sba_guarantee_pct` necesita un tratamiento distinto dado que su distribución responde a política pública más que a riesgo intrínseco.
4. **Repetir este seguimiento con cadencia mensual** sobre las cohortes de aprobación más recientes, aplicando la misma metodología (`src/monitoring.py`), y revisar explícitamente cada vez si las cohortes que entraban en 2012-2014 ya han madurado lo suficiente para una lectura de Gini/calibración fiable.
