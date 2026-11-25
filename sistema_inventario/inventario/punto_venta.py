"""
Gestión del inventario, descuentos y bitácora.

Guarda los datos en una carpeta (por defecto ./punto_venta):
  - inventario_actual.csv : productos y existencias; se descuentan con cada venta
  - descuentos.csv        : descuentos aplicados por producto (sugerido o manual)
  - movimientos.csv       : bitácora de altas, bajas, cambios, entradas y descuentos
  - modelo.joblib         : modelo de clasificación entrenado con el inventario inicial

El precio de venta es precio_venta menos el descuento aplicado. Las recomendaciones
no se aplican solas: hay que aplicarlas desde la sección Descuentos.
"""
import os
from datetime import datetime

import joblib
import pandas as pd

from . import clasificacion, datos, descuentos

CARPETA_DEFAULT = "punto_venta"
LIMITE_DESCUENTO_MANUAL = 60.0  # porcentaje máximo que se puede aplicar a mano

COLUMNAS_MOVIMIENTOS = ["fecha", "id_producto", "tipo", "detalle"]
COLUMNAS_DESCUENTOS = ["id_producto", "descuento_pct", "origen", "fecha"]
CAMPOS_EDITABLES = [
    "nombre", "categoria", "precio_compra", "precio_venta", "existencia_actual",
    "unidades_vendidas", "dias_periodo", "dias_desde_ultima_venta",
]


def _rutas(carpeta):
    return {
        "inventario": os.path.join(carpeta, "inventario_actual.csv"),
        "modelo": os.path.join(carpeta, "modelo.joblib"),
        "movimientos": os.path.join(carpeta, "movimientos.csv"),
        "aplicados": os.path.join(carpeta, "descuentos.csv"),
    }


def _ahora():
    return datetime.now().isoformat(timespec="seconds")


def esta_inicializado(carpeta=CARPETA_DEFAULT):
    r = _rutas(carpeta)
    return os.path.exists(r["inventario"]) and os.path.exists(r["modelo"])


def inicializar(df, modelo, carpeta=CARPETA_DEFAULT):
    """Carga un inventario nuevo. Reemplaza el inventario y borra la bitácora y los descuentos."""
    os.makedirs(carpeta, exist_ok=True)
    r = _rutas(carpeta)
    df.to_csv(r["inventario"], index=False)
    joblib.dump(modelo, r["modelo"])
    pd.DataFrame(columns=COLUMNAS_MOVIMIENTOS).to_csv(r["movimientos"], index=False)
    pd.DataFrame(columns=COLUMNAS_DESCUENTOS).to_csv(r["aplicados"], index=False)


# ---------- Lectura ----------

def cargar_inventario(carpeta=CARPETA_DEFAULT):
    return pd.read_csv(_rutas(carpeta)["inventario"], dtype={"id_producto": str})


def cargar_movimientos(carpeta=CARPETA_DEFAULT):
    r = _rutas(carpeta)
    if not os.path.exists(r["movimientos"]):
        return pd.DataFrame(columns=COLUMNAS_MOVIMIENTOS)
    return pd.read_csv(r["movimientos"], dtype={"id_producto": str})


def cargar_aplicados(carpeta=CARPETA_DEFAULT):
    r = _rutas(carpeta)
    if not os.path.exists(r["aplicados"]):
        return pd.DataFrame(columns=COLUMNAS_DESCUENTOS)
    return pd.read_csv(r["aplicados"], dtype={"id_producto": str})


def _guardar_aplicados(carpeta, aplicados):
    aplicados.to_csv(_rutas(carpeta)["aplicados"], index=False)


def registrar_movimiento(carpeta, id_producto, tipo, detalle):
    """Agrega una línea a la bitácora de movimientos."""
    r = _rutas(carpeta)
    fila = pd.DataFrame([{
        "fecha": _ahora(),
        "id_producto": id_producto,
        "tipo": tipo,
        "detalle": detalle,
    }])
    fila.to_csv(r["movimientos"], mode="a", header=not os.path.exists(r["movimientos"]), index=False)


# ---------- Recomendaciones y precios ----------

def recomendaciones_actuales(carpeta, max_descuento, margen_minimo):
    """Descuentos sugeridos según el inventario actual."""
    r = _rutas(carpeta)
    inv = cargar_inventario(carpeta)
    modelo = joblib.load(r["modelo"])
    etiquetas = clasificacion.predecir(modelo, inv)
    etiquetas[inv["existencia_actual"] == 0] = "Normal"
    return descuentos.generar_recomendaciones(inv, etiquetas, max_descuento, margen_minimo)


def _precio_final(precio_venta, pct):
    return round(float(precio_venta) * (1 - float(pct) / 100), 2)


def tabla_precios(carpeta=CARPETA_DEFAULT):
    """Precio de cobro de cada producto: precio de lista menos el descuento aplicado."""
    inv = cargar_inventario(carpeta)
    aplicados = cargar_aplicados(carpeta).set_index("id_producto")
    tabla = inv[[
        "id_producto", "nombre", "categoria", "existencia_actual",
        "precio_compra", "precio_venta",
    ]].copy()
    tabla["descuento_pct"] = tabla["id_producto"].map(aplicados["descuento_pct"]).fillna(0.0)
    tabla["origen"] = tabla["id_producto"].map(aplicados["origen"]).fillna("")
    tabla["precio_final"] = (tabla["precio_venta"] * (1 - tabla["descuento_pct"] / 100)).round(2)
    return tabla


def tabla_descuentos(carpeta, max_descuento, margen_minimo):
    """Precios actuales junto con la sugerencia del sistema para cada producto."""
    tabla = tabla_precios(carpeta)
    recs = recomendaciones_actuales(carpeta, max_descuento, margen_minimo).set_index("id_producto")
    tabla["etiqueta"] = tabla["id_producto"].map(recs["etiqueta"])
    tabla["sugerido_pct"] = tabla["id_producto"].map(recs["descuento_pct"]).fillna(0.0)
    tabla["motivo"] = tabla["id_producto"].map(recs["motivo"])
    return tabla


# ---------- Descuentos ----------

def aplicar_sugerencias(carpeta, max_descuento, margen_minimo):
    """Aplica el descuento sugerido a todos los productos candidatos. Devuelve cuántos fueron."""
    recs = recomendaciones_actuales(carpeta, max_descuento, margen_minimo)
    candidatos = recs[recs["descuento_pct"] > 0]
    ids = candidatos["id_producto"].astype(str).tolist()

    aplicados = cargar_aplicados(carpeta)
    aplicados = aplicados[~aplicados["id_producto"].isin(ids)]
    nuevos = pd.DataFrame({
        "id_producto": ids,
        "descuento_pct": candidatos["descuento_pct"].astype(float).round(2).tolist(),
        "origen": "sugerido",
        "fecha": _ahora(),
    }, columns=COLUMNAS_DESCUENTOS)
    _guardar_aplicados(carpeta, pd.concat([aplicados, nuevos], ignore_index=True))

    if ids:
        registrar_movimiento(carpeta, "*", "descuento", f"Sugerencias aplicadas a {len(ids)} productos")
    return len(ids)


def guardar_descuento(carpeta, id_producto, pct, sugerido_pct=None):
    """
    Fija el descuento de un producto. pct = 0 quita el descuento.
    Devuelve el origen guardado: 'sugerido' o 'manual'. Lanza ValueError si no es válido.
    """
    pid = str(id_producto)
    inv = cargar_inventario(carpeta)
    fila = inv[inv["id_producto"] == pid]
    if fila.empty:
        raise ValueError(f"El producto {pid} no existe.")
    pct = float(pct)
    nombre = fila["nombre"].iloc[0]
    if pct < 0 or pct > LIMITE_DESCUENTO_MANUAL:
        raise ValueError(
            f"{nombre}: el descuento debe estar entre 0 y {LIMITE_DESCUENTO_MANUAL:.0f}%."
        )
    precio_venta = float(fila["precio_venta"].iloc[0])
    costo = float(fila["precio_compra"].iloc[0])
    if pct > 0 and _precio_final(precio_venta, pct) < costo:
        raise ValueError(
            f"{nombre}: con {pct:.1f}% el precio quedaría en "
            f"${_precio_final(precio_venta, pct):.2f}, por debajo del costo (${costo:.2f})."
        )

    antes = float(cargar_aplicados(carpeta).set_index("id_producto")["descuento_pct"].get(pid, 0.0))
    origen = "sugerido" if sugerido_pct is not None and abs(pct - float(sugerido_pct)) < 0.05 else "manual"

    aplicados = cargar_aplicados(carpeta)
    aplicados = aplicados[aplicados["id_producto"] != pid]
    if pct > 0:
        aplicados = pd.concat([aplicados, pd.DataFrame([{
            "id_producto": pid, "descuento_pct": round(pct, 2), "origen": origen, "fecha": _ahora(),
        }])], ignore_index=True)
    _guardar_aplicados(carpeta, aplicados)

    if round(antes, 2) != round(pct, 2):
        registrar_movimiento(carpeta, pid, "descuento", f"{antes:.1f}% → {pct:.1f}% ({origen})")
    return origen


# ---------- Gestión de inventario ----------

def _iguales(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def guardar_cambios(nuevo, carpeta=CARPETA_DEFAULT):
    """
    Reemplaza el inventario con la tabla editada. Valida antes de guardar y
    registra en la bitácora cada alta, baja y cambio. Quita los descuentos de
    productos eliminados o que quedarían por debajo del costo.
    Devuelve el número de movimientos registrados.
    """
    actual = cargar_inventario(carpeta)
    nuevo = nuevo.copy()
    nuevo["id_producto"] = nuevo["id_producto"].astype(str).str.strip()
    if nuevo["id_producto"].isin(["", "nan", "None"]).any():
        raise ValueError("Todos los productos deben tener id_producto.")
    if nuevo["id_producto"].duplicated().any():
        duplicados = nuevo.loc[nuevo["id_producto"].duplicated(), "id_producto"].tolist()
        raise ValueError(f"Hay id_producto repetidos: {', '.join(duplicados)}")

    limpio, problemas = datos.validar(nuevo)
    if len(limpio) != len(nuevo):
        raise ValueError("No se guardó. Revisa estos datos: " + "; ".join(problemas))

    limpio = limpio.set_index("id_producto")
    act = actual.set_index("id_producto")
    movimientos = 0

    for p in limpio.index.difference(act.index):
        registrar_movimiento(carpeta, p, "alta", f"Nuevo producto: {limpio.loc[p, 'nombre']}")
        movimientos += 1
    for p in act.index.difference(limpio.index):
        registrar_movimiento(carpeta, p, "baja", f"Producto eliminado: {act.loc[p, 'nombre']}")
        movimientos += 1
    for p in limpio.index.intersection(act.index):
        for campo in CAMPOS_EDITABLES:
            antes, despues = act.loc[p, campo], limpio.loc[p, campo]
            if not _iguales(antes, despues):
                if campo == "existencia_actual":
                    tipo = "existencia"
                elif campo.startswith("precio"):
                    tipo = "precio"
                else:
                    tipo = "edicion"
                registrar_movimiento(carpeta, p, tipo, f"{campo}: {antes} → {despues}")
                movimientos += 1

    resultado = limpio.reset_index().reindex(columns=actual.columns)
    resultado.to_csv(_rutas(carpeta)["inventario"], index=False)

    # Revisar descuentos vigentes frente a los nuevos datos
    validos = []
    for _, f in cargar_aplicados(carpeta).iterrows():
        pid = f["id_producto"]
        if pid not in limpio.index:
            motivo = "producto eliminado"
        elif _precio_final(limpio.loc[pid, "precio_venta"], f["descuento_pct"]) < float(limpio.loc[pid, "precio_compra"]):
            motivo = "quedaría por debajo del costo"
        else:
            validos.append(f.to_dict())
            continue
        registrar_movimiento(carpeta, pid, "descuento", f"Descuento de {f['descuento_pct']}% quitado: {motivo}")
        movimientos += 1
    _guardar_aplicados(carpeta, pd.DataFrame(validos, columns=COLUMNAS_DESCUENTOS))

    return movimientos


def registrar_entrada(id_producto, cantidad, costo_unitario=None, carpeta=CARPETA_DEFAULT):
    """Suma existencias por una entrada de mercancía. Opcionalmente actualiza el costo unitario."""
    if cantidad <= 0:
        raise ValueError("La cantidad debe ser mayor a cero.")
    pid = str(id_producto)
    inv = cargar_inventario(carpeta)
    mascara = inv["id_producto"] == pid
    if not mascara.any():
        raise ValueError(f"El producto {pid} no existe.")

    detalle = f"+{int(cantidad)} unidades"
    inv.loc[mascara, "existencia_actual"] += int(cantidad)
    if costo_unitario:
        antes = inv.loc[mascara, "precio_compra"].iloc[0]
        inv.loc[mascara, "precio_compra"] = float(costo_unitario)
        detalle += f"; costo: {antes} → {costo_unitario}"

    inv.to_csv(_rutas(carpeta)["inventario"], index=False)
    registrar_movimiento(carpeta, pid, "entrada", detalle)
