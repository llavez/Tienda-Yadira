"""Pruebas del punto de venta. Ejecutar con: python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inventario import clasificacion, datos, punto_venta  # noqa: E402


class TestPuntoVenta(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        df, _ = datos.validar(datos.generar_ejemplo(n=150, semilla=3))
        train, val, _ = clasificacion.dividir(df, 3)
        modelo, _ = clasificacion.entrenar(train, val, 3)
        punto_venta.inicializar(df, modelo, self.tmp)
        self.inv = punto_venta.cargar_inventario(self.tmp)

    def test_venta_descuenta_existencia_y_guarda_ticket(self):
        pid = str(self.inv["id_producto"].iloc[0])
        antes = int(self.inv.set_index("id_producto").loc[pid, "existencia_actual"])
        ticket, total = punto_venta.registrar_venta({pid: 1}, carpeta=self.tmp)
        despues = int(punto_venta.cargar_inventario(self.tmp).set_index("id_producto").loc[pid, "existencia_actual"])
        self.assertEqual(despues, antes - 1)
        self.assertEqual(len(ticket), 1)
        self.assertEqual(len(punto_venta.cargar_ventas(self.tmp)), 1)
        self.assertAlmostEqual(total, round(float(ticket["precio_unitario"].iloc[0]), 2), places=2)

    def test_stock_insuficiente_no_modifica_nada(self):
        pid = str(self.inv["id_producto"].iloc[0])
        antes = int(self.inv.set_index("id_producto").loc[pid, "existencia_actual"])
        with self.assertRaises(ValueError):
            punto_venta.registrar_venta({pid: antes + 1}, carpeta=self.tmp)
        despues = int(punto_venta.cargar_inventario(self.tmp).set_index("id_producto").loc[pid, "existencia_actual"])
        self.assertEqual(despues, antes)
        self.assertEqual(len(punto_venta.cargar_ventas(self.tmp)), 0)

    def test_carrito_vacio_falla(self):
        with self.assertRaises(ValueError):
            punto_venta.registrar_venta({}, carpeta=self.tmp)

    def test_producto_inexistente_falla(self):
        with self.assertRaises(ValueError):
            punto_venta.registrar_venta({"NOEXISTE": 1}, carpeta=self.tmp)

    def test_resumen_del_dia(self):
        pid = str(self.inv["id_producto"].iloc[0])
        punto_venta.registrar_venta({pid: 2}, carpeta=self.tmp)
        resumen = punto_venta.resumen_del_dia(self.tmp)
        self.assertEqual(resumen["tickets"], 1)
        self.assertGreater(resumen["ventas"], 0)


if __name__ == "__main__":
    unittest.main()
