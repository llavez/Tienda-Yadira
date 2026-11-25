"""
Recomendación de descuentos.

Reglas:
  - Baja rotación:       descuento base.
  - Exceso de inventario: descuento base que crece con los días de cobertura extra.
  - Nunca se supera el tope máximo de descuento.
  - El precio final debe dejar al menos el margen mínimo sobre el costo.
"""
import pandas as pd

from .clasificacion import UMBRAL_COBERTURA_DIAS, construir_caracteristicas

MAX_DESCUENTO = 0.30           # tope absoluto (30%)
MARGEN_MINIMO = 0.10           # el precio final debe cubrir costo + 10%
DESCUENTO_BASE_BAJA = 0.10
DESCUENTO_BASE_EXCESO = 0.10
INCREMENTO_POR_PERIODO = 0.05  # por cada periodo de UMBRAL_COBERTURA_DIAS extra


def calcular_descuento(etiqueta, cobertura_dias, precio_compra, precio_venta,
                       max_descuento=MAX_DESCUENTO, margen_minimo=MARGEN_MINIMO):
    """Devuelve (descuento como fracción, motivo)."""
    if etiqueta == "Normal":
        return 0.0, "No requiere descuento"

    if etiqueta == "Baja rotación":
        propuesto = DESCUENTO_BASE_BAJA
    else:  # Exceso de inventario
        extra = max(0.0, cobertura_dias - UMBRAL_COBERTURA_DIAS) / UMBRAL_COBERTURA_DIAS
        propuesto = DESCUENTO_BASE_EXCESO + INCREMENTO_POR_PERIODO * extra

    # Descuento máximo que todavía respeta el margen mínimo sobre el costo
    limite_margen = max(0.0, 1 - precio_compra * (1 + margen_minimo) / precio_venta)
    final = round(min(propuesto, max_descuento, limite_margen), 2)

    if final <= 0:
        return 0.0, "Sin descuento: el margen no lo permite"
    if final < propuesto:
        return final, "Descuento limitado por tope o margen mínimo"
    return final, "Descuento según rotación o exceso"


def generar_recomendaciones(df, etiquetas, max_descuento=MAX_DESCUENTO,
                            margen_minimo=MARGEN_MINIMO):
    """Tabla de recomendaciones ordenada por capital inmovilizado (mayor primero)."""
    x = construir_caracteristicas(df)
    filas = []
    for idx, fila in df.iterrows():
        etiqueta = etiquetas.loc[idx]
        descuento, motivo = calcular_descuento(
            etiqueta,
            x.loc[idx, "cobertura_dias"],
            fila["precio_compra"],
            fila["precio_venta"],
            max_descuento,
            margen_minimo,
        )
        filas.append({
            "id_producto": fila["id_producto"],
            "nombre": fila["nombre"],
            "categoria": fila["categoria"],
            "etiqueta": etiqueta,
            "candidato_descuento": etiqueta != "Normal",
            "existencia_actual": fila["existencia_actual"],
            "capital_inmovilizado": round(fila["existencia_actual"] * fila["precio_compra"], 2),
            "precio_venta": fila["precio_venta"],
            "descuento_pct": round(descuento * 100, 1),
            "precio_sugerido": round(fila["precio_venta"] * (1 - descuento), 2),
            "motivo": motivo,
        })
    return pd.DataFrame(filas).sort_values("capital_inmovilizado", ascending=False)
