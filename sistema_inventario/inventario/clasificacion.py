"""
Clasificación de productos.

Etiquetas (según el protocolo):
  - Normal
  - Baja rotación
  - Exceso de inventario

"Candidato a descuento" es cualquier producto cuya etiqueta no sea Normal.
Las métricas de precisión y recall se calculan sobre esa decisión binaria.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from .datos import ETIQUETAS_VALIDAS

# Reglas de negocio (ajustables según la política de la tienda)
UMBRAL_ROTACION_DIARIA = 0.2   # menos de 1 venta cada 5 días => baja rotación
UMBRAL_DIAS_SIN_VENTA = 30     # más de 30 días sin vender => baja rotación
UMBRAL_COBERTURA_DIAS = 90     # más de 3 meses de existencia al ritmo actual => exceso
CAP_COBERTURA = 9999           # valor máximo cuando no hay ventas

COLUMNAS_MODELO = [
    "rotacion_diaria",
    "cobertura_dias",
    "dias_desde_ultima_venta",
    "existencia_actual",
    "precio_venta",
]


def construir_caracteristicas(df):
    """Calcula las variables que usa el modelo a partir de los datos crudos."""
    x = pd.DataFrame(index=df.index)
    x["rotacion_diaria"] = df["unidades_vendidas"] / df["dias_periodo"]
    rot = x["rotacion_diaria"]
    cobertura = np.where(
        rot > 0,
        df["existencia_actual"] / np.where(rot > 0, rot, 1),
        CAP_COBERTURA,
    )
    x["cobertura_dias"] = np.minimum(cobertura, CAP_COBERTURA)
    x["dias_desde_ultima_venta"] = df["dias_desde_ultima_venta"]
    x["existencia_actual"] = df["existencia_actual"]
    x["precio_venta"] = df["precio_venta"]
    return x


def etiquetar_regla(df):
    """Clasificación por reglas de negocio. Sirve como respaldo y para generar datos de prueba."""
    x = construir_caracteristicas(df)
    etiqueta = pd.Series("Normal", index=df.index)
    baja = (x["rotacion_diaria"] < UMBRAL_ROTACION_DIARIA) | (
        df["dias_desde_ultima_venta"] > UMBRAL_DIAS_SIN_VENTA
    )
    exceso = x["cobertura_dias"] > UMBRAL_COBERTURA_DIAS
    etiqueta[baja] = "Baja rotación"
    etiqueta[exceso] = "Exceso de inventario"  # tiene prioridad sobre baja rotación
    etiqueta[df["existencia_actual"] == 0] = "Normal"  # nada que descontar
    return etiqueta


def dividir(df, semilla=42):
    """Partición estratificada 70% entrenamiento / 15% validación / 15% prueba."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["etiqueta"], random_state=semilla
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["etiqueta"], random_state=semilla
    )
    return train, val, test


def entrenar(train, val, semilla=42):
    """
    Entrena un Random Forest probando varias profundidades.
    Se elige la de mejor F1 macro en validación (el conjunto de prueba no se usa aquí).
    """
    x_tr = construir_caracteristicas(train)[COLUMNAS_MODELO]
    y_tr = train["etiqueta"]
    x_val = construir_caracteristicas(val)[COLUMNAS_MODELO]
    y_val = val["etiqueta"]

    mejor_modelo, mejor_f1 = None, -1.0
    for profundidad in [3, 5, 8, None]:
        modelo = RandomForestClassifier(
            n_estimators=200,
            max_depth=profundidad,
            class_weight="balanced",
            random_state=semilla,
        )
        modelo.fit(x_tr, y_tr)
        f1 = f1_score(y_val, modelo.predict(x_val), average="macro", zero_division=0)
        if f1 > mejor_f1:
            mejor_modelo, mejor_f1 = modelo, f1
    return mejor_modelo, float(mejor_f1)


def predecir(modelo, df):
    """Devuelve la etiqueta predicha por el modelo para cada producto."""
    x = construir_caracteristicas(df)[COLUMNAS_MODELO]
    return pd.Series(modelo.predict(x), index=df.index)


def evaluar(modelo, df):
    """Métricas del protocolo: exactitud por clase, precisión y recall de candidatos a descuento."""
    y_real = df["etiqueta"]
    y_pred = predecir(modelo, df)

    candidato_real = y_real != "Normal"
    candidato_pred = y_pred != "Normal"

    matriz = confusion_matrix(y_real, y_pred, labels=ETIQUETAS_VALIDAS)
    return {
        "accuracy_clases": float(accuracy_score(y_real, y_pred)),
        "precision_candidatos_descuento": float(
            precision_score(candidato_real, candidato_pred, zero_division=0)
        ),
        "recall_candidatos_descuento": float(
            recall_score(candidato_real, candidato_pred, zero_division=0)
        ),
        "matriz_confusion": {
            "labels": ETIQUETAS_VALIDAS,
            "valores": [[int(v) for v in fila] for fila in matriz],
        },
        "n_evaluados": int(len(df)),
    }
