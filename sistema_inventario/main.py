"""
Uso:
  python main.py                          # usa un dataset de ejemplo generado
  python main.py --datos inventario.csv   # usa tus datos (CSV o Excel)
"""
import argparse
import json
import os

import joblib
import pandas as pd

from inventario import clasificacion, datos, descuentos


def main():
    parser = argparse.ArgumentParser(description="Gestión de inventario y descuentos automáticos")
    parser.add_argument("--datos", help="CSV o Excel con el inventario")
    parser.add_argument("--salida", default="resultados", help="Carpeta de resultados")
    parser.add_argument("--max-descuento", type=float, default=descuentos.MAX_DESCUENTO)
    parser.add_argument("--margen-minimo", type=float, default=descuentos.MARGEN_MINIMO)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    # 1. Datos
    if args.datos:
        crudo = datos.cargar(args.datos)
    else:
        print("No se indicó archivo: se usa un dataset de ejemplo generado.")
        crudo = datos.generar_ejemplo(semilla=args.semilla)

    df, problemas = datos.validar(crudo)
    print(f"Registros válidos: {len(df)}")
    for p in problemas:
        print(f"  - {p}")

    # 2. Etiquetas de entrenamiento: las del encargado; la regla completa los faltantes
    regla = clasificacion.etiquetar_regla(df)
    if datos.COLUMNA_ETIQUETA not in df.columns:
        df[datos.COLUMNA_ETIQUETA] = regla
    else:
        df[datos.COLUMNA_ETIQUETA] = df[datos.COLUMNA_ETIQUETA].fillna(regla)

    # 3. Modelo de IA con partición 70/15/15
    train, val, test = clasificacion.dividir(df, args.semilla)
    modelo, f1_val = clasificacion.entrenar(train, val, args.semilla)
    metricas = clasificacion.evaluar(modelo, test)
    metricas["f1_macro_validacion"] = f1_val
    metricas["particion"] = {"train": len(train), "validacion": len(val), "prueba": len(test)}

    print("\nMétricas en conjunto de prueba:")
    print(f"  Exactitud por clase:           {metricas['accuracy_clases']:.2%}")
    print(f"  Precisión (candidatos):        {metricas['precision_candidatos_descuento']:.2%}")
    print(f"  Recall (candidatos):           {metricas['recall_candidatos_descuento']:.2%}")

    # 4. Recomendaciones sobre todo el inventario actual
    etiquetas_ia = clasificacion.predecir(modelo, df)
    recs = descuentos.generar_recomendaciones(
        df, etiquetas_ia, args.max_descuento, args.margen_minimo
    )

    candidatos = recs[recs["candidato_descuento"]]
    print(f"\nProductos candidatos a descuento: {len(candidatos)} de {len(recs)}")
    print(f"Capital inmovilizado en candidatos: ${candidatos['capital_inmovilizado'].sum():,.2f}")
    if len(candidatos):
        print(f"Descuento promedio recomendado:     {candidatos['descuento_pct'].mean():.1f}%")

    # 5. Guardar resultados
    recs.to_csv(os.path.join(args.salida, "recomendaciones.csv"), index=False)
    with open(os.path.join(args.salida, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    joblib.dump(modelo, os.path.join(args.salida, "modelo.joblib"))
    print(f"\nResultados guardados en: {args.salida}/")


if __name__ == "__main__":
    main()
