# 🔍 Detección de Fraude en Transacciones Financieras

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11-9B59B6?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Estado-Activo-brightgreen)

Pipeline completo de detección de fraude en transacciones con tarjeta de crédito. Incluye técnicas de balanceo de clases, detección de anomalías no supervisada, clasificación supervisada y optimización de umbral por costo de negocio.

---

## 🎯 Problema de Negocio

En fintech boliviana y LATAM, el fraude en transacciones digitales es uno de los problemas más costosos. El desafío técnico no es solo el modelado — es el **desbalance extremo**: solo 0.17% de las transacciones son fraude.

Un modelo naive que diga siempre "no fraude" tiene **99.83% de accuracy**. Es completamente inútil. Este proyecto demuestra por qué las métricas correctas importan tanto como los algoritmos.

---

## 🧠 Enfoque Técnico

### Dos estrategias complementarias

| Enfoque | Modelo | Cuándo usarlo |
|---|---|---|
| **No supervisado** | Isolation Forest | Cuando no tienes suficientes ejemplos de fraude etiquetados |
| **Supervisado** | Random Forest + SMOTE | Cuando tienes ejemplos etiquetados aunque sean pocos |

### ¿Por qué SMOTE y no simple oversampling?

SMOTE genera ejemplos sintéticos de fraude **interpolando** entre ejemplos reales en el espacio de features, en lugar de simplemente duplicar. Esto reduce el overfitting a ejemplos específicos y mejora la generalización.

```python
# Duplicar (malo): el modelo memoriza los mismos ejemplos
# SMOTE (mejor): crea variabilidad sintética realista
from imblearn.over_sampling import SMOTE
smote = SMOTE(k_neighbors=5)  # Interpola entre 5 vecinos más cercanos
```

### ¿Por qué Average Precision en vez de AUC-ROC?

Con desbalance extremo, AUC-ROC puede ser engañosamente alto. Un modelo mediocre puede tener AUC=0.95 con 0.17% de fraudes. **Average Precision (área bajo la curva Precision-Recall) es más honesto** porque penaliza fuertemente los falsos positivos en la clase minoritaria.

---

## 📊 Dataset

**Credit Card Fraud Detection** — Kaggle / ULB Machine Learning Group

- 284,807 transacciones europeas (septiembre 2013)
- 492 fraudes (0.172%)
- Features V1-V28: componentes PCA (anonimizadas por privacidad)
- Features originales: `Time` y `Amount`

📥 [Descargar en Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → guardar como `datos/creditcard.csv`

---

## 📁 Estructura

```
deteccion-fraude-transacciones/
├── deteccion_fraude.ipynb     # Análisis exploratorio + comparación de modelos
├── src/
│   └── modelo_fraude.py       # Pipeline de producción — entrenar y predecir
├── datos/                     # CSV de Kaggle (no incluido — ver descarga)
├── modelos/                   # Modelos serializados (.pkl) — generados al entrenar
├── img/                       # Gráficas exportadas
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Instalación

```bash
git clone https://github.com/Serius69/deteccion-fraude-transacciones
cd deteccion-fraude-transacciones

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p datos modelos img

# Descargar creditcard.csv de Kaggle → colocar en datos/
```

**Opción A — Notebook exploratorio:**
```bash
jupyter lab deteccion_fraude.ipynb
```

**Opción B — Script de producción:**
```bash
# Entrenamiento con umbral automático
python src/modelo_fraude.py --datos datos/creditcard.csv --umbral auto

# Entrenamiento con umbral fijo
python src/modelo_fraude.py --datos datos/creditcard.csv --umbral 0.35
```

**Predecir una transacción nueva:**
```python
from src.modelo_fraude import predecir_transaccion

transaccion = {
    'V1': -1.36, 'V2': -0.07, 'V3': 2.53, 'V4': 1.37,
    # ... V5-V28 ...
    'Amount': 149.62, 'Time': 3600
}
resultado = predecir_transaccion(transaccion)
print(resultado)
# {'score': 0.0423, 'es_fraude': False, 'riesgo': '🟢 BAJO'}
```

---

## 📈 Resultados Esperados

| Modelo | AUC-ROC | Average Precision |
|---|---|---|
| Isolation Forest | ~0.95 | ~0.28 |
| LR + SMOTE | ~0.97 | ~0.71 |
| **RF + SMOTE** | **~0.98** | **~0.85** |

> Los resultados exactos varían por split aleatorio.

---

## 💡 Decisiones de Negocio Implementadas

**Umbral óptimo ≠ 0.5:** El umbral se calcula minimizando el costo total:

```
Costo total = FN × monto_fraude_promedio + FP × $2
```

Rechazar una transacción legítima cuesta ~$2 (atención al cliente). No detectar un fraude cuesta el monto completo de la transacción. El modelo ajusta el umbral para minimizar este costo real.

---

## 🔗 Proyectos Relacionados

- [💳 Scoring de Crédito Fintech](https://github.com/Serius69/scoring-credito-fintech-latam) — Clasificación de riesgo crediticio
- [💱 Dólar Paralelo Bolivia — ARIMA](https://github.com/Serius69/dolar-paralelo-bolivia-arima) — Riesgo de mercado

---

## 👤 Autor

**Sergio** — Data Scientist en Finanzas | [github.com/Serius69](https://github.com/Serius69)
