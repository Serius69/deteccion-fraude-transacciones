"""
src/modelo_fraude.py
─────────────────────────────────────────────────────────────────────────────
Pipeline de producción para detección de fraude.
Versión script .py del notebook — más limpio para deployment.

Uso:
    python src/modelo_fraude.py --datos datos/creditcard.csv --umbral auto
    python src/modelo_fraude.py --datos datos/creditcard.csv --umbral 0.35
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
SEED         = 42
TEST_SIZE    = 0.20
COSTO_FP     = 2.0       # $ por falso positivo (transacción legítima rechazada)
MODELS_DIR   = 'modelos'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('img', exist_ok=True)


# ─── FUNCIONES ────────────────────────────────────────────────────────────────

def cargar_y_preprocesar(ruta_csv: str):
    """Carga el dataset y escala Amount/Time."""
    print(f"📂 Cargando {ruta_csv}...")
    df = pd.read_csv(ruta_csv)

    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])

    features = [c for c in df.columns if c.startswith('V')] + ['Amount_scaled', 'Time_scaled']
    X = df[features]
    y = df['Class']

    tasa_fraude = y.mean()
    print(f"   Shape: {df.shape} | Tasa fraude: {tasa_fraude*100:.4f}%")
    return X, y, df, scaler, features


def entrenar(X_train, y_train):
    """Entrena Random Forest con SMOTE."""
    print("\n🔧 Entrenando Random Forest + SMOTE...")
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=5)),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            random_state=SEED,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    print("   ✅ Entrenamiento completado")
    return pipeline


def evaluar(pipeline, X_test, y_test, montos_test, umbral='auto'):
    """Evalúa el modelo y calcula umbral óptimo si se pide."""
    proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    ap  = average_precision_score(y_test, proba)

    print(f"\n📊 Métricas (umbral=0.5 por defecto):")
    print(f"   AUC-ROC:           {auc:.4f}")
    print(f"   Average Precision: {ap:.4f}")

    if umbral == 'auto':
        umbral = _calcular_umbral_optimo(proba, y_test, montos_test)

    pred_opt = (proba >= umbral).astype(int)
    print(f"\n🎯 Resultados con umbral óptimo ({umbral:.3f}):")
    print(classification_report(y_test, pred_opt, target_names=['Legítima', 'Fraude']))

    cm = confusion_matrix(y_test, pred_opt)
    tn, fp, fn, tp = cm.ravel()
    print(f"   Fraudes detectados:  {tp} / {tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"   Falsas alarmas:      {fp} / {tn+fp} ({fp/(tn+fp)*100:.2f}%)")

    return proba, umbral, auc, ap


def _calcular_umbral_optimo(proba, y_test, montos_test, costo_fp=COSTO_FP):
    """Minimiza costo total: FN × monto_fraude + FP × costo_fp."""
    print("\n⚙️  Calculando umbral óptimo por costo asimétrico...")
    umbrales = np.arange(0.01, 0.99, 0.005)
    costos   = []

    for u in umbrales:
        pred = (proba >= u).astype(int)
        cm   = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        monto_fn = montos_test[(y_test == 1).values & (pred == 0)].mean() if fn > 0 else 0
        costos.append(fn * monto_fn + fp * costo_fp)

    umbral_opt = umbrales[np.argmin(costos)]
    print(f"   Umbral óptimo encontrado: {umbral_opt:.3f} (costo mínimo: ${min(costos):,.2f})")
    return umbral_opt


def guardar_modelo(pipeline, scaler, features, umbral, auc, ap):
    """Serializa el modelo y metadata."""
    ruta = os.path.join(MODELS_DIR, 'modelo_fraude.pkl')
    joblib.dump({
        'pipeline': pipeline,
        'scaler':   scaler,
        'features': features,
        'umbral':   umbral,
        'metricas': {'auc': auc, 'ap': ap}
    }, ruta)
    print(f"\n💾 Modelo guardado en {ruta}")


def predecir_transaccion(datos_dict: dict, modelo_path='modelos/modelo_fraude.pkl'):
    """
    Predice si una transacción es fraude.
    
    Args:
        datos_dict: dict con claves V1-V28 + Amount + Time
    
    Returns:
        dict con 'score', 'es_fraude', 'riesgo'
    """
    artefacto = joblib.load(modelo_path)
    pipeline  = artefacto['pipeline']
    scaler    = artefacto['scaler']
    features  = artefacto['features']
    umbral    = artefacto['umbral']

    df_tx = pd.DataFrame([datos_dict])
    df_tx['Amount_scaled'] = scaler.transform(df_tx[['Amount']])
    df_tx['Time_scaled']   = scaler.transform(df_tx[['Time']])

    score     = pipeline.predict_proba(df_tx[features])[0][1]
    es_fraude = score >= umbral

    if score < 0.1:
        riesgo = '🟢 BAJO'
    elif score < umbral:
        riesgo = '🟡 MEDIO'
    else:
        riesgo = '🔴 ALTO — POSIBLE FRAUDE'

    return {'score': round(score, 4), 'es_fraude': es_fraude, 'riesgo': riesgo}


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline de detección de fraude')
    parser.add_argument('--datos',   default='datos/creditcard.csv', help='Ruta al CSV')
    parser.add_argument('--umbral',  default='auto', help='Umbral (float o "auto")')
    args = parser.parse_args()

    umbral = float(args.umbral) if args.umbral != 'auto' else 'auto'

    # Pipeline completo
    X, y, df, scaler, features = cargar_y_preprocesar(args.datos)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    montos_test = df.loc[y_test.index, 'Amount']

    pipeline = entrenar(X_train, y_train)
    proba, umbral_usado, auc, ap = evaluar(pipeline, X_test, y_test, montos_test, umbral)
    guardar_modelo(pipeline, scaler, features, umbral_usado, auc, ap)

    print("\n✅ Pipeline completado exitosamente")
    print(f"   Para predecir nuevas transacciones: from src.modelo_fraude import predecir_transaccion")
