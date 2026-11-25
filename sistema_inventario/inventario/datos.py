"""Carga, validación y generación de datos de inventario."""
import numpy as np
import pandas as pd

COLUMNAS_REQUERIDAS = [
    "id_producto",
    "nombre",
    "categoria",
    "precio_compra",
    "precio_venta",
    "existencia_actual",
    "unidades_vendidas",
    "dias_periodo",
    "dias_desde_ultima_venta",
]
COLUMNAS_NUMERICAS = [
    "precio_compra",
    "precio_venta",
    "existencia_actual",
    "unidades_vendidas",
    "dias_periodo",
    "dias_desde_ultima_venta",
]
# Columna opcional: etiqueta asignada por el encargado (para entrenar el modelo)
COLUMNA_ETIQUETA = "etiqueta"
ETIQUETAS_VALIDAS = ["Normal", "Baja rotación", "Exceso de inventario"]


def cargar(ruta):
    """Lee un archivo CSV o Excel con los registros de inventario."""
    if str(ruta).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(ruta)
    return pd.read_csv(ruta)


def validar(df):
    """Limpia el dataset. Devuelve (df_limpio, lista_de_problemas encontrados)."""
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    problemas = []

    # 1. Productos duplicados
    duplicados = df.duplicated(subset="id_producto", keep="first")
    if duplicados.any():
        problemas.append(f"{duplicados.sum()} registros duplicados eliminados")
    df = df[~duplicados].copy()

    # 2. Valores numéricos mal formados, vacíos o negativos
    df[COLUMNAS_NUMERICAS] = df[COLUMNAS_NUMERICAS].apply(pd.to_numeric, errors="coerce")
    malos = (
        df[COLUMNAS_REQUERIDAS].isna().any(axis=1)
        | (df[COLUMNAS_NUMERICAS] < 0).any(axis=1)
        | (df["dias_periodo"] == 0)
        | (df["precio_venta"] == 0)
    )
    if malos.any():
        problemas.append(
            f"{malos.sum()} registros incompletos o con valores inválidos eliminados"
        )
    df = df[~malos].copy()

    # 3. Etiquetas no reconocidas: se tratan como sin etiqueta
    if COLUMNA_ETIQUETA in df.columns:
        no_validas = df[COLUMNA_ETIQUETA].notna() & ~df[COLUMNA_ETIQUETA].isin(ETIQUETAS_VALIDAS)
        if no_validas.any():
            problemas.append(
                f"{no_validas.sum()} etiquetas no reconocidas (se tratarán como sin etiqueta)"
            )
            df.loc[no_validas, COLUMNA_ETIQUETA] = np.nan

    return df.reset_index(drop=True), problemas


def generar_ejemplo(n=600, semilla=42, ruta=None):
    """
    Genera un inventario sintético para probar el sistema.
    Las etiquetas simulan el criterio del encargado: se parten de la regla
    de negocio y se cambia un 10% para representar juicio humano no uniforme.
    Se agregan duplicados a propósito para probar la validación.
    """
    from .clasificacion import etiquetar_regla

    rng = np.random.default_rng(semilla)
    categorias = ["Abarrotes", "Bebidas", "Limpieza", "Electrónica", "Ropa", "Papelería"]

    precio_compra = rng.uniform(15, 600, n).round(2)
    precio_venta = (precio_compra * rng.uniform(1.15, 2.0, n)).round(2)
    # Ventas promedio por día: unas categorías rotan mucho, otras poco
    tasa_diaria = rng.choice([0.02, 0.1, 0.5, 2.0, 4.0], size=n, p=[0.2, 0.25, 0.25, 0.2, 0.1])
    unidades_vendidas = rng.poisson(tasa_diaria * 90)
    dias_periodo = np.full(n, 90)
    existencia_actual = rng.integers(0, 250, n)
    dias_desde_ultima_venta = np.where(
        unidades_vendidas > 0, rng.integers(0, 90, n), dias_periodo
    )

    df = pd.DataFrame({
        "id_producto": [f"P{i:04d}" for i in range(1, n + 1)],
        "nombre": [f"Producto {i}" for i in range(1, n + 1)],
        "categoria": rng.choice(categorias, n),
        "precio_compra": precio_compra,
        "precio_venta": precio_venta,
        "existencia_actual": existencia_actual,
        "unidades_vendidas": unidades_vendidas,
        "dias_periodo": dias_periodo,
        "dias_desde_ultima_venta": dias_desde_ultima_venta,
    })

    etiquetas = etiquetar_regla(df).to_numpy(copy=True)
    cambiar = rng.random(n) < 0.10
    etiquetas[cambiar] = rng.choice(ETIQUETAS_VALIDAS, cambiar.sum())
    df[COLUMNA_ETIQUETA] = etiquetas

    df = pd.concat([df, df.head(3)], ignore_index=True)  # duplicados intencionales

    if ruta:
        df.to_csv(ruta, index=False)
    return df
