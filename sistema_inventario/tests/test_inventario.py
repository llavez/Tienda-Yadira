"""Pruebas. Ejecutar con: python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inventario import clasificacion, datos, descuentos  # noqa: E402


def fila(**kwargs):
    base = {
        "id_producto": "X1", "nombre": "Test", "categoria": "Abarrotes",
        "precio_compra": 50.0, "precio_venta": 100.0, "existencia_actual": 10,
        "unidades_vendidas": 30, "dias_periodo": 90, "dias_desde_ultima_venta": 2,
    }
    base.update(kwargs)
    return pd.DataFrame([base])


class TestValidacion(unittest.TestCase):
    def test_elimina_duplicados_y_registros_invalidos(self):
        df = pd.concat([
            fila(id_producto="A"),
            fila(id_producto="A"),                 # duplicado
            fila(id_producto="B", precio_venta=0), # precio inválido
            fila(id_producto="C", existencia_actual=None),
        ], ignore_index=True)
        limpio, problemas = datos.validar(df)
        self.assertEqual(list(limpio["id_producto"]), ["A"])
        self.assertTrue(any("duplicados" in p for p in problemas))

    def test_falta_columna_lanza_error(self):
        with self.assertRaises(ValueError):
            datos.validar(pd.DataFrame({"id_producto": ["A"]}))


class TestClasificacion(unittest.TestCase):
    def test_baja_rotacion(self):
        df = fila(unidades_vendidas=3, dias_desde_ultima_venta=10, existencia_actual=2)  # cobertura 60 días
        self.assertEqual(clasificacion.etiquetar_regla(df).iloc[0], "Baja rotación")

    def test_exceso_de_inventario(self):
        # 1 venta en 90 días con 200 unidades => cobertura enorme
        df = fila(unidades_vendidas=1, dias_desde_ultima_venta=5, existencia_actual=200)
        self.assertEqual(clasificacion.etiquetar_regla(df).iloc[0], "Exceso de inventario")

    def test_sin_existencia_es_normal(self):
        df = fila(unidades_vendidas=0, dias_desde_ultima_venta=90, existencia_actual=0)
        self.assertEqual(clasificacion.etiquetar_regla(df).iloc[0], "Normal")


class TestDescuentos(unittest.TestCase):
    def test_normal_no_tiene_descuento(self):
        d, _ = descuentos.calcular_descuento("Normal", 10, 50, 100)
        self.assertEqual(d, 0.0)

    def test_nunca_supera_el_tope(self):
        d, _ = descuentos.calcular_descuento("Exceso de inventario", 9999, 10, 100,
                                             max_descuento=0.30)
        self.assertLessEqual(d, 0.30)

    def test_respeta_margen_minimo(self):
        # costo 80, venta 100, margen mínimo 10% => precio final >= 88 => descuento <= 12%
        d, _ = descuentos.calcular_descuento("Baja rotación", 10, 80, 100,
                                             max_descuento=0.30, margen_minimo=0.10)
        self.assertLessEqual(d, 0.12)
        self.assertGreaterEqual(100 * (1 - d), 80 * 1.10 - 1e-9)

    def test_margen_insuficiente_no_descuenta(self):
        d, motivo = descuentos.calcular_descuento("Baja rotación", 10, 99, 100)
        self.assertEqual(d, 0.0)
        self.assertIn("margen", motivo)


class TestPipeline(unittest.TestCase):
    def test_extremo_a_extremo(self):
        df, _ = datos.validar(datos.generar_ejemplo(n=300, semilla=7))
        df["etiqueta"] = clasificacion.etiquetar_regla(df)
        train, val, test = clasificacion.dividir(df, 7)
        self.assertAlmostEqual(len(train) / len(df), 0.70, delta=0.02)

        modelo, _ = clasificacion.entrenar(train, val, 7)
        metricas = clasificacion.evaluar(modelo, test)
        for clave in ["accuracy_clases", "precision_candidatos_descuento",
                      "recall_candidatos_descuento"]:
            self.assertGreaterEqual(metricas[clave], 0.0)
            self.assertLessEqual(metricas[clave], 1.0)

        recs = descuentos.generar_recomendaciones(df, clasificacion.predecir(modelo, df))
        self.assertEqual(len(recs), len(df))
        self.assertTrue((recs["descuento_pct"] <= descuentos.MAX_DESCUENTO * 100).all())


if __name__ == "__main__":
    unittest.main()
