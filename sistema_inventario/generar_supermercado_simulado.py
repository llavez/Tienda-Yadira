"""
Genera un inventario SIMULADO de supermercado con nombres de productos reales.

Los nombres y tamaños son de productos de venta común; los precios, existencias
y ventas son INVENTADOS. Sirve para probar el sistema y la interfaz, no para
validar exactitud ni para tomar decisiones de negocio.

Uso:
  python generar_supermercado_simulado.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inventario.clasificacion import etiquetar_regla  # noqa: E402
from inventario.datos import COLUMNA_ETIQUETA, ETIQUETAS_VALIDAS  # noqa: E402

SALIDA = "inventario_supermercado_SIMULADO.csv"
SEMILLA = 2026

# (nombre, categoría, precio de compra aproximado en MXN, inventado)
PRODUCTOS = [
    ("Coca-Cola 600 ml", "Bebidas", 14.00),
    ("Coca-Cola 2.5 L", "Bebidas", 34.00),
    ("Sprite 600 ml", "Bebidas", 13.00),
    ("Agua Ciel 1 L", "Bebidas", 9.00),
    ("Jugo Del Valle Durazno 1 L", "Bebidas", 22.00),
    ("Cerveza Corona 355 ml", "Bebidas", 17.00),
    ("Gatorade Naranja 600 ml", "Bebidas", 19.00),
    ("Té Lipton Limón 1 L", "Bebidas", 21.00),
    ("Sabritas Original 45 g", "Botanas", 11.00),
    ("Sabritas Adobadas 45 g", "Botanas", 11.00),
    ("Doritos Nacho 62 g", "Botanas", 13.50),
    ("Ruffles Queso 47 g", "Botanas", 11.50),
    ("Cacahuates Japoneses Kreelo 120 g", "Botanas", 15.00),
    ("Galletas Emperador Chocolate 100 g", "Botanas", 12.00),
    ("Galletas Marías Gamesa 170 g", "Botanas", 14.00),
    ("Chocolate Carlos V 40 g", "Botanas", 8.00),
    ("Maseca Harina de Maíz 1 kg", "Abarrotes", 17.00),
    ("Arroz Sós Grano Largo 1 kg", "Abarrotes", 24.00),
    ("Frijol Negro Verde Valle 900 g", "Abarrotes", 28.00),
    ("Azúcar Estándar Zulka 1 kg", "Abarrotes", 26.00),
    ("Aceite Nutrioli 900 ml", "Abarrotes", 42.00),
    ("Sal Refinada La Fina 1 kg", "Abarrotes", 9.00),
    ("Atún Dolores en Agua 140 g", "Abarrotes", 15.00),
    ("Sardina Calmex en Tomate 155 g", "Abarrotes", 18.00),
    ("Salsa Valentina 370 ml", "Abarrotes", 16.00),
    ("Mayonesa Helmans 390 g", "Abarrotes", 34.00),
    ("Pasta Barilla Spaghetti 500 g", "Abarrotes", 23.00),
    ("Sopa Maruchan Camarón 64 g", "Abarrotes", 10.00),
    ("Mole Doña María 160 g", "Abarrotes", 29.00),
    ("Café Nescafé Clásico 100 g", "Abarrotes", 78.00),
    ("Leche Lala Entera 1 L", "Lácteos", 21.00),
    ("Leche Alpura Deslactosada 1 L", "Lácteos", 23.00),
    ("Yoghurt Danone Fresa 125 g", "Lácteos", 9.50),
    ("Queso Oaxaca Lala 400 g", "Lácteos", 62.00),
    ("Crema Lala para Batir 250 ml", "Lácteos", 25.00),
    ("Mantequilla Gloria 90 g", "Lácteos", 19.00),
    ("Huevo Blanco San Juan 18 pzas", "Lácteos", 58.00),
    ("Pan Bimbo Blanco Grande", "Panadería", 42.00),
    ("Pan Bimbo Integral", "Panadería", 46.00),
    ("Tortillas de Harina Mission 10 pzas", "Panadería", 35.00),
    ("Bolillo Pieza", "Panadería", 3.50),
    ("Marinela Gansito 2 pzas", "Panadería", 14.00),
    ("Pastel Bimbo Chocolate", "Panadería", 39.00),
    ("Detergente Ariel Polvo 1 kg", "Limpieza", 58.00),
    ("Cloro Cloralex 1 L", "Limpieza", 17.00),
    ("Suavitel Fresh 850 ml", "Limpieza", 39.00),
    ("Jabón Zote Rosa 400 g", "Limpieza", 13.00),
    ("Fabuloso Lavanda 1 L", "Limpieza", 27.00),
    ("Pinol Original 828 ml", "Limpieza", 36.00),
    ("Estropajo Scotch-Brite Pieza", "Limpieza", 12.00),
    ("Papel Higiénico Kleenex 4 rollos", "Higiene", 36.00),
    ("Papel Higiénico Regio 12 rollos", "Higiene", 84.00),
    ("Servilletas Kleenex 100 pzas", "Higiene", 22.00),
    ("Shampoo Head & Shoulders 400 ml", "Higiene", 69.00),
    ("Shampoo Sedal Keratina 340 ml", "Higiene", 49.00),
    ("Pasta Dental Colgate 100 ml", "Higiene", 32.00),
    ("Desodorante Rexona Clinical 50 ml", "Higiene", 41.00),
    ("Rastrillos Bic 3 pzas", "Higiene", 27.00),
    ("Pañales Huggies Etapa 4 (20 pzas)", "Higiene", 189.00),
    ("Toallas Sanitarias Saba Ultra 10 pzas", "Higiene", 34.00),
    ("Pollo Entero Bachoco 1 kg", "Carnes", 74.00),
    ("Salchicha Fud Viena 250 g", "Carnes", 29.00),
    ("Jamón Fud Virginia 200 g", "Carnes", 36.00),
    ("Chorizo Coronado Rojo 250 g", "Carnes", 48.00),
    ("Plátano Tabasco kg", "Frutas y verduras", 19.00),
    ("Manzana Roja kg", "Frutas y verduras", 38.00),
    ("Jitomate Saladette kg", "Frutas y verduras", 26.00),
    ("Cebolla Blanca kg", "Frutas y verduras", 21.00),
    ("Aguacate Hass Pieza", "Frutas y verduras", 16.00),
    ("Limón Pieza", "Frutas y verduras", 1.50),
    ("Pilas Duracell AA 4 pzas", "Varios", 54.00),
    ("Velas Aromáticas Pieza", "Varios", 15.00),
    ("Bolsas de Plástico Asa Pieza", "Varios", 0.80),
]

CANTIDAD_PRODUCTOS = len(PRODUCTOS)


SUFIJOS_VARIANTES = [
    "Paquete 3 pzas", "Paquete 6 pzas", "Presentación familiar",
    "Presentación chica", "Edición económica",
]


def generar(variantes=False):
    """
    Genera el inventario simulado. Con variantes=True agrega 3 presentaciones
    extra por cada producto base (unos 290 productos en total).
    """
    rng = np.random.default_rng(SEMILLA)
    productos = list(PRODUCTOS)
    if variantes:
        rng_var = np.random.default_rng(SEMILLA + 1)
        for nombre, categoria, costo_base in PRODUCTOS:
            for sufijo in rng_var.choice(SUFIJOS_VARIANTES, 3, replace=False):
                factor = rng_var.uniform(1.6, 4.0)
                productos.append((f"{nombre} - {sufijo}", categoria, round(costo_base * factor, 2)))

    n = len(productos)
    nombres = [p[0] for p in productos]
    categorias = [p[1] for p in productos]
    costo = np.array([p[2] for p in productos], dtype=float)

    # Margen de venta inventado: entre 20% y 60%
    precio_venta = (costo * rng.uniform(1.20, 1.60, n)).round(2)

    # Ritmo de venta diario inventado: algunos productos rotan mucho, otros poco
    tasa = rng.choice([0.05, 0.3, 1.0, 4.0, 12.0], size=n, p=[0.15, 0.2, 0.3, 0.25, 0.1])
    dias_periodo = 90
    vendidas = rng.poisson(tasa * dias_periodo)

    # Existencia inventada: suele ser proporcional al ritmo de venta, con sobrestock aleatorio
    existencia = np.maximum(0, (tasa * rng.uniform(5, 60, n)).round()).astype(int)
    sobrestock = rng.random(n) < 0.2
    existencia[sobrestock] += rng.integers(100, 300, sobrestock.sum())

    dias_sin_venta = np.where(vendidas > 0, rng.integers(0, dias_periodo, n), dias_periodo)

    df = pd.DataFrame({
        "id_producto": [f"SIM{i:03d}" for i in range(1, n + 1)],
        "nombre": nombres,
        "categoria": categorias,
        "precio_compra": costo,
        "precio_venta": precio_venta,
        "existencia_actual": existencia,
        "unidades_vendidas": vendidas,
        "dias_periodo": dias_periodo,
        "dias_desde_ultima_venta": dias_sin_venta,
    })

    # Etiqueta del "encargado" simulada: regla de negocio con 10% de desacuerdo
    etiquetas = etiquetar_regla(df).to_numpy(copy=True)
    cambiar = rng.random(n) < 0.10
    etiquetas[cambiar] = rng.choice(ETIQUETAS_VALIDAS, cambiar.sum())
    df[COLUMNA_ETIQUETA] = etiquetas

    df["origen"] = "SIMULADO"
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genera inventario simulado de supermercado")
    parser.add_argument("--extenso", action="store_true", help="Incluye presentaciones extra (~290 productos)")
    parser.add_argument("--salida", default=None, help="Nombre del CSV de salida")
    args = parser.parse_args()

    salida = args.salida or ("inventario_supermercado_SIMULADO_extenso.csv" if args.extenso else SALIDA)
    df = generar(variantes=args.extenso)
    df.to_csv(salida, index=False, encoding="utf-8")
    print(f"Generados {len(df)} productos simulados en {salida}")
