"""Pruebas de edición del inventario. Ejecutar con: python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inventario import clasificacion, datos, punto_venta  # noqa: E402


class TestGestionInventario(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        df, _ = datos.validar(datos.generar_ejemplo(n=120, semilla=5))
        train, val, _ = clasificacion.dividir(df, 5)
        modelo, _ = clasificacion.entrenar(train, val, 5)
        punto_venta.inicializar(df, modelo, self.tmp)
        self.inv = punto_venta.cargar_inventario(self.tmp)
        self.pid = str(self.inv["id_producto"].iloc[0])

    def _leer(self):
        return punto_venta.cargar_inventario(self.tmp)

    def test_cambiar_precio_y_existencia_se_guarda_y_registra(self):
        editado = self.inv.copy()
        editado.loc[0, "precio_venta"] = editado.loc[0, "precio_venta"] + 5
        editado.loc[0, "existencia_actual"] = 999
        n = punto_venta.guardar_cambios(editado, self.tmp)
        self.assertEqual(n, 2)
        fila = self._leer().set_index("id_producto").loc[self.pid]
        self.assertEqual(int(fila["existencia_actual"]), 999)
        bitacora = punto_venta.cargar_movimientos(self.tmp)
        self.assertEqual(set(bitacora["tipo"]), {"precio", "existencia"})

    def test_agregar_producto(self):
        nuevo = pd.concat([self.inv, pd.DataFrame([{
            "id_producto": "NUEVO1", "nombre": "Producto nuevo", "categoria": "Abarrotes",
            "precio_compra": 10.0, "precio_venta": 15.0, "existencia_actual": 20,
            "unidades_vendidas": 0, "dias_periodo": 90, "dias_desde_ultima_venta": 90,
        }])], ignore_index=True)
        punto_venta.guardar_cambios(nuevo, self.tmp)
        self.assertEqual(len(self._leer()), len(self.inv) + 1)
        self.assertIn("alta", set(punto_venta.cargar_movimientos(self.tmp)["tipo"]))

    def test_eliminar_producto(self):
        punto_venta.guardar_cambios(self.inv.iloc[1:].copy(), self.tmp)
        self.assertEqual(len(self._leer()), len(self.inv) - 1)
        self.assertIn("baja", set(punto_venta.cargar_movimientos(self.tmp)["tipo"]))

    def test_precio_negativo_se_rechaza_sin_cambios(self):
        editado = self.inv.copy()
        editado.loc[0, "precio_venta"] = -1
        with self.assertRaises(ValueError):
            punto_venta.guardar_cambios(editado, self.tmp)
        self.assertEqual(len(self._leer()), len(self.inv))
        self.assertEqual(len(punto_venta.cargar_movimientos(self.tmp)), 0)

    def test_id_repetido_se_rechaza(self):
        editado = pd.concat([self.inv, self.inv.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            punto_venta.guardar_cambios(editado, self.tmp)

    def test_entrada_suma_existencia_y_actualiza_costo(self):
        antes = int(self._leer().set_index("id_producto").loc[self.pid, "existencia_actual"])
        punto_venta.registrar_entrada(self.pid, 10, costo_unitario=12.5, carpeta=self.tmp)
        fila = self._leer().set_index("id_producto").loc[self.pid]
        self.assertEqual(int(fila["existencia_actual"]), antes + 10)
        self.assertEqual(float(fila["precio_compra"]), 12.5)
        self.assertIn("entrada", set(punto_venta.cargar_movimientos(self.tmp)["tipo"]))

    def test_entrada_cantidad_invalida(self):
        with self.assertRaises(ValueError):
            punto_venta.registrar_entrada(self.pid, 0, carpeta=self.tmp)


if __name__ == "__main__":
    unittest.main()
