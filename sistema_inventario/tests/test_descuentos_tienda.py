"""Pruebas de descuentos aplicados a los precios de venta."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inventario import clasificacion, datos, punto_venta  # noqa: E402


class TestDescuentosTienda(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        df, _ = datos.validar(datos.generar_ejemplo(n=150, semilla=9))
        train, val, _ = clasificacion.dividir(df, 9)
        modelo, _ = clasificacion.entrenar(train, val, 9)
        punto_venta.inicializar(df, modelo, self.tmp)
        self.inv = punto_venta.cargar_inventario(self.tmp).set_index("id_producto")

    def test_sin_descuento_aplicado_usa_precio_de_lista(self):
        pid = self.inv.index[0]
        precio = punto_venta.tabla_precios(self.tmp).set_index("id_producto").loc[pid, "precio_final"]
        self.assertAlmostEqual(float(precio), round(float(self.inv.loc[pid, "precio_venta"]), 2), places=2)

    def test_aplicar_sugerencias_y_cobrar_con_descuento(self):
        n = punto_venta.aplicar_sugerencias(self.tmp, 0.3, 0.1)
        self.assertGreater(n, 0)
        aplicados = punto_venta.cargar_aplicados(self.tmp)
        self.assertEqual(len(aplicados), n)
        self.assertTrue((aplicados["origen"] == "sugerido").all())

        pid = aplicados.iloc[0]["id_producto"]
        pct = float(aplicados.iloc[0]["descuento_pct"])
        precio_lista = float(self.inv.loc[pid, "precio_venta"])
        precio_final = punto_venta.tabla_precios(self.tmp).set_index("id_producto").loc[pid, "precio_final"]
        self.assertAlmostEqual(float(precio_final), round(precio_lista * (1 - pct / 100), 2), places=2)

    def test_descuento_manual_se_guarda_como_manual(self):
        pid = self.inv.index[0]
        precio = float(self.inv.loc[pid, "precio_venta"])
        origen = punto_venta.guardar_descuento(self.tmp, pid, 5.0, sugerido_pct=20.0)
        self.assertEqual(origen, "manual")
        self.assertEqual(float(punto_venta.tabla_precios(self.tmp).set_index("id_producto").loc[pid, "descuento_pct"]), 5.0)
        self.assertAlmostEqual(
            float(punto_venta.tabla_precios(self.tmp).set_index("id_producto").loc[pid, "precio_final"]),
            round(precio * 0.95, 2), places=2,
        )

    def test_descuento_igual_a_sugerido_queda_como_sugerido(self):
        pid = self.inv.index[0]
        origen = punto_venta.guardar_descuento(self.tmp, pid, 12.0, sugerido_pct=12.0)
        self.assertEqual(origen, "sugerido")

    def test_descuento_que_baja_del_costo_se_rechaza(self):
        pid = self.inv.index[0]
        with self.assertRaises(ValueError):
            punto_venta.guardar_descuento(self.tmp, pid, 99.0)
        self.assertEqual(len(punto_venta.cargar_aplicados(self.tmp)), 0)

    def test_descuento_fuera_de_limite_se_rechaza(self):
        pid = self.inv.index[0]
        with self.assertRaises(ValueError):
            punto_venta.guardar_descuento(self.tmp, pid, -1)
        with self.assertRaises(ValueError):
            punto_venta.guardar_descuento(self.tmp, pid, punto_venta.LIMITE_DESCUENTO_MANUAL + 1)

    def test_cero_quita_el_descuento(self):
        pid = self.inv.index[0]
        punto_venta.guardar_descuento(self.tmp, pid, 5.0)
        punto_venta.guardar_descuento(self.tmp, pid, 0.0)
        self.assertEqual(len(punto_venta.cargar_aplicados(self.tmp)), 0)

    def test_eliminar_producto_quita_su_descuento(self):
        pid = self.inv.index[0]
        punto_venta.guardar_descuento(self.tmp, pid, 5.0)
        inv = punto_venta.cargar_inventario(self.tmp)
        punto_venta.guardar_cambios(inv[inv["id_producto"] != pid], self.tmp)
        self.assertEqual(len(punto_venta.cargar_aplicados(self.tmp)), 0)


if __name__ == "__main__":
    unittest.main()
