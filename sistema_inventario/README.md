# Sistema inteligente para la gestión de inventario y aplicación automática de descuentos

Proyecto integrador de Técnicas de Inteligencia Artificial (FIMAZ-UAS).

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Con datos de ejemplo generados automáticamente
python main.py

# Con datos propios (CSV o Excel)
python main.py --datos inventario.csv --salida resultados --max-descuento 0.25 --margen-minimo 0.10
```

Salidas en la carpeta indicada con `--salida`:

- `recomendaciones.csv`: por producto, etiqueta, si es candidato a descuento, descuento sugerido, precio sugerido y capital inmovilizado.
- `metricas.json`: exactitud, precisión, recall y matriz de confusión sobre el conjunto de prueba.
- `modelo.joblib`: modelo entrenado.

## Formato de los datos

Columnas requeridas:

| Columna | Descripción |
|---|---|
| `id_producto` | Identificador único |
| `nombre`, `categoria` | Descripción del producto |
| `precio_compra`, `precio_venta` | Costo y precio de venta unitarios |
| `existencia_actual` | Unidades en inventario hoy |
| `unidades_vendidas` | Unidades vendidas en el periodo |
| `dias_periodo` | Duración del periodo analizado (p. ej. 90) |
| `dias_desde_ultima_venta` | Días transcurridos desde la última venta |

Columna opcional:

| Columna | Descripción |
|---|---|
| `etiqueta` | `Normal`, `Baja rotación` o `Exceso de inventario`, asignada por el encargado. Si se incluye, el modelo aprende su criterio. |

## Cómo se relaciona con el protocolo

**Clasificación.** El sistema asigna una de tres etiquetas. "Candidato a descuento" es cualquier producto que no sea Normal. Se calculan dos variables de negocio: rotación diaria (unidades vendidas / días del periodo) y cobertura en días (existencia / rotación diaria).

- Exceso de inventario: cobertura mayor a 90 días.
- Baja rotación: rotación menor a 0.2 unidades por día, o más de 30 días sin venta.

El modelo de IA es un Random Forest entrenado con partición estratificada 70% / 15% / 15%. La profundidad se elige con el conjunto de validación; el de prueba se usa solo para reportar métricas.

**Métricas técnicas.** Exactitud por clase, precisión y recall de candidatos a descuento, tal como se definieron en la sección 5 del protocolo. Un descuento innecesario se refleja en la precisión; un producto que necesita acción sin detectar, en el recall.

**Descuentos.** Las reglas de la sección 6 se implementan así:

- Tope máximo de descuento (por defecto 30%).
- Margen mínimo sobre el costo (por defecto 10%): el precio final nunca cae por debajo de costo × 1.10.
- El descuento sube con la cobertura excedente en el caso de exceso de inventario.

**Indicadores organizacionales.** La reducción de productos de baja rotación y el tiempo de revisión se miden en la tienda, comparando contra el baseline (sección 4). El sistema genera el dato de capital inmovilizado para dar seguimiento.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

## Limitaciones conocidas

- Las reglas y los umbrales son supuestos iniciales. Deben validarse con el encargado de la tienda.
- El dataset de ejemplo es sintético. Para validar la exactitud técnica (criterio de 80%) hay que comparar contra conteo físico real.
- El dataset público de Kaggle "Retail Store Inventory and Demand Forecasting" tiene otro esquema de columnas. Hay que renombrarlas o agregar un adaptador antes de usarlo.
- Si una clase tiene menos de 2 registros, la partición estratificada fallará. Se requiere un volumen mínimo de datos.
- El sistema recomienda descuentos; la tienda decide cuáles aplicar.

## Dataset de prueba simulado

`inventario_supermercado_SIMULADO.csv` contiene 73 productos con nombres reales de supermercado. Los precios, existencias y ventas son **inventados**; la columna `origen` lo indica. Sirve para probar el sistema y la interfaz, no para validar exactitud ni para decisiones de negocio.

Para regenerarlo: `python generar_supermercado_simulado.py`


## Gestión de inventario

La sección **Inventario** permite:

- **Editar la tabla** directamente: precios, existencias, nombres, categorías. Al guardar, los datos se validan antes; si algo no cuadra (precio negativo, id repetido, campo vacío), no se guarda nada y se muestra el motivo.
- **Agregar productos** usando la última fila vacía de la tabla.
- **Eliminar productos** seleccionando la fila y pulsando Supr.
- **Registrar entradas de mercancía**, que suman existencias y, opcionalmente, actualizan el costo unitario.
- **Consultar la bitácora** (`punto_venta/movimientos.csv`), con cada alta, baja, cambio de precio, cambio de existencia y entrada.

Los descuentos recomendados se recalculan con los cambios al instante. El modelo de clasificación no se reentrena al editar el inventario; para eso hay que volver a cargar el inventario desde el archivo original.

## Datasets de prueba

- `inventario_supermercado_SIMULADO.csv`: 73 productos.
- `inventario_supermercado_SIMULADO_extenso.csv`: 292 productos, incluye presentaciones extra (paquete, familiar, económica).

Ambos son simulados (`origen = SIMULADO`). Para regenerar: `python generar_supermercado_simulado.py` o `python generar_supermercado_simulado.py --extenso`.


## Descuentos

La sección **Descuentos** separa la recomendación del descuento real que se cobra:

- **Sugerido:** lo que calcula el modelo y las reglas (tope y margen). Se muestra en la tabla, pero no se cobra hasta aplicarlo.
- **Aplicado:** el descuento que realmente se usa en los precios de venta. Se guarda en `punto_venta/descuentos.csv`.

Formas de trabajar:

- **Aplicar todas las sugerencias:** pone en vigor el descuento sugerido a todos los candidatos.
- **Editar a mano:** cambia el porcentaje de cualquier producto en la tabla y pulsa **Guardar descuentos editados**. Pon 0 para quitar el descuento.

Reglas para descuentos manuales:

- Entre 0% y 60%.
- El precio final nunca puede quedar por debajo del costo (`precio_compra`).
- Si un producto se elimina o cambia de precio y su descuento deja de ser válido, el descuento se quita y queda registrado en la bitácora.

Cada cambio de descuento se registra en la bitácora de movimientos.
