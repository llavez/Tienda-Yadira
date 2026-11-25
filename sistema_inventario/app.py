"""
Interfaz gráfica: análisis, descuentos e inventario.

Ejecutar con:
  streamlit run app.py
"""
import io
import os

import pandas as pd
import streamlit as st

from inventario import clasificacion, datos, descuentos, punto_venta

BASE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(BASE, punto_venta.CARPETA_DEFAULT)
ARCHIVO_INICIAL = "inventario_supermercado_SIMULADO_extenso.csv"

st.set_page_config(page_title="Inventario y descuentos", layout="wide")
st.title("Gestión de inventario y descuentos automáticos")

# ---------- Barra lateral ----------
with st.sidebar:
    seccion = st.radio("Sección", ["Análisis", "Descuentos", "Inventario"])

    st.header("Datos")
    archivo = st.file_uploader("Inventario (CSV o Excel)", type=["csv", "xlsx", "xls"])
    ruta_local = st.text_input(
        "O ruta de un archivo local",
        value=ARCHIVO_INICIAL,
        help="Ruta relativa a la carpeta del proyecto, o ruta absoluta. Borra el texto para usar el dataset generado.",
    )

    st.header("Parámetros")
    max_desc = st.slider("Tope máximo de descuento sugerido (%)", 5, 60, 30) / 100
    margen = st.slider("Margen mínimo sobre costo (%)", 0, 50, 10) / 100
    semilla = st.number_input("Semilla aleatoria", value=42, step=1)


@st.cache_data(show_spinner="Validando datos y entrenando el modelo...")
def procesar(contenido, nombre, semilla):
    """Valida datos, etiqueta y entrena el modelo. Se guarda en caché mientras no cambie el archivo."""
    if contenido is None:
        crudo = datos.generar_ejemplo(semilla=semilla)
    else:
        buf = io.BytesIO(contenido)
        buf.name = nombre
        crudo = pd.read_excel(buf) if nombre.lower().endswith((".xlsx", ".xls")) else pd.read_csv(buf)

    df, problemas = datos.validar(crudo)

    regla = clasificacion.etiquetar_regla(df)
    if datos.COLUMNA_ETIQUETA not in df.columns:
        df[datos.COLUMNA_ETIQUETA] = regla
    else:
        df[datos.COLUMNA_ETIQUETA] = df[datos.COLUMNA_ETIQUETA].fillna(regla)

    train, val, test = clasificacion.dividir(df, semilla)
    modelo, f1_val = clasificacion.entrenar(train, val, semilla)
    metricas = clasificacion.evaluar(modelo, test)
    metricas["f1_macro_validacion"] = f1_val
    return df, problemas, modelo, metricas


try:
    if archivo is not None:
        contenido, nombre = archivo.getvalue(), archivo.name
    elif ruta_local.strip():
        nombre = ruta_local.strip()
        ruta = nombre if os.path.isabs(nombre) else os.path.join(BASE, nombre)
        with open(ruta, "rb") as f:
            contenido = f.read()
    else:
        contenido, nombre = None, "ejemplo.csv"
    df, problemas, modelo, metricas = procesar(contenido, nombre, int(semilla))
except Exception as e:
    st.error(f"No se pudo procesar el archivo: {e}")
    st.stop()

# Al abrir la app, si todavía no hay inventario guardado, se carga el que está en uso
if not punto_venta.esta_inicializado(C):
    punto_venta.inicializar(df, modelo, C)

with st.sidebar:
    st.header("Inventario en uso")
    st.caption("Cargar este inventario reemplaza el actual y borra descuentos y bitácora.")
    if st.button("Cargar este inventario"):
        punto_venta.inicializar(df, modelo, C)
        st.success("Inventario cargado.")


# ---------- Análisis ----------
def pagina_analisis():
    st.subheader("1. Datos")
    st.write(f"Registros válidos: **{len(df)}**")
    for p in problemas:
        st.warning(p)
    if not problemas:
        st.success("Sin problemas de calidad detectados.")

    st.subheader("2. Desempeño del modelo (conjunto de prueba)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Exactitud por clase", f"{metricas['accuracy_clases']:.1%}")
    c2.metric("Precisión (candidatos)", f"{metricas['precision_candidatos_descuento']:.1%}")
    c3.metric("Recall (candidatos)", f"{metricas['recall_candidatos_descuento']:.1%}")

    with st.expander("Matriz de confusión"):
        mc = metricas["matriz_confusion"]
        matriz = pd.DataFrame(
            mc["valores"],
            index=[f"Real: {l}" for l in mc["labels"]],
            columns=[f"Predicho: {l}" for l in mc["labels"]],
        )
        st.dataframe(matriz)

    etiquetas_ia = clasificacion.predecir(modelo, df)
    recs = descuentos.generar_recomendaciones(df, etiquetas_ia, max_desc, margen)
    candidatos = recs[recs["candidato_descuento"]]

    st.subheader("3. Recomendaciones de descuento")
    k1, k2, k3 = st.columns(3)
    k1.metric("Candidatos a descuento", f"{len(candidatos)} de {len(recs)}")
    k2.metric("Capital inmovilizado", f"${candidatos['capital_inmovilizado'].sum():,.0f}")
    k3.metric(
        "Descuento promedio",
        f"{candidatos['descuento_pct'].mean():.1f}%" if len(candidatos) else "—",
    )

    st.markdown("**Capital inmovilizado por categoría (solo candidatos)**")
    if len(candidatos):
        st.bar_chart(candidatos.groupby("categoria")["capital_inmovilizado"].sum())
    else:
        st.info("No hay productos candidatos con estos parámetros.")

    f1, f2 = st.columns(2)
    etiquetas_sel = f1.multiselect(
        "Filtrar por etiqueta",
        sorted(recs["etiqueta"].unique()),
        default=sorted(recs["etiqueta"].unique()),
    )
    categorias_sel = f2.multiselect(
        "Filtrar por categoría",
        sorted(recs["categoria"].unique()),
        default=sorted(recs["categoria"].unique()),
    )
    vista = recs[recs["etiqueta"].isin(etiquetas_sel) & recs["categoria"].isin(categorias_sel)]
    st.dataframe(vista, width="stretch", hide_index=True)

    st.download_button(
        "Descargar recomendaciones (CSV)",
        data=recs.to_csv(index=False).encode("utf-8"),
        file_name="recomendaciones.csv",
        mime="text/csv",
    )


# ---------- Descuentos ----------
def pagina_descuentos():
    if not punto_venta.esta_inicializado():
        st.info("Carga primero un inventario con el botón de la barra lateral.")
        return

    if "version_desc" not in st.session_state:
        st.session_state.version_desc = 0
    if "mensaje_desc" in st.session_state:
        st.success(st.session_state.pop("mensaje_desc"))

    tabla = punto_venta.tabla_descuentos(C, max_desc, margen)
    sugeridos = tabla[tabla["sugerido_pct"] > 0]
    pendientes = sugeridos[(sugeridos["descuento_pct"] - sugeridos["sugerido_pct"]).abs() >= 0.05]
    con_descuento = int((tabla["descuento_pct"] > 0).sum())

    d1, d2, d3 = st.columns(3)
    d1.metric("Productos con descuento", con_descuento)
    d2.metric("Sugerencias por aplicar", len(pendientes))
    d3.metric("Tope sugerido", f"{max_desc * 100:.0f}%")

    st.caption(
        "Las sugerencias son recomendaciones del modelo; no se cobran hasta aplicarlas. "
        "Puedes aplicar todas de una vez o editar el porcentaje de cada producto en la tabla. "
        "Pon 0 para quitar un descuento. El máximo manual es "
        f"{punto_venta.LIMITE_DESCUENTO_MANUAL:.0f}% y el precio nunca puede quedar por debajo del costo."
    )

    if st.button("Aplicar todas las sugerencias", type="primary", disabled=pendientes.empty):
        n = punto_venta.aplicar_sugerencias(C, max_desc, margen)
        st.session_state.mensaje_desc = f"Sugerencias aplicadas a {n} productos."
        st.session_state.version_desc += 1
        st.rerun()

    vista = tabla[[
        "id_producto", "nombre", "categoria", "existencia_actual",
        "precio_compra", "precio_venta", "etiqueta", "sugerido_pct",
        "descuento_pct", "origen", "motivo",
    ]].rename(columns={"descuento_pct": "descuento_aplicado_pct"})
    original = vista.set_index("id_producto")["descuento_aplicado_pct"]
    sugerido = vista.set_index("id_producto")["sugerido_pct"]

    editado = st.data_editor(
        vista,
        hide_index=True,
        width="stretch",
        key=f"editor_desc_{st.session_state.version_desc}",
        disabled=[c for c in vista.columns if c != "descuento_aplicado_pct"],
        column_config={
            "descuento_aplicado_pct": st.column_config.NumberColumn(
                "Descuento aplicado (%)",
                min_value=0.0,
                max_value=punto_venta.LIMITE_DESCUENTO_MANUAL,
                step=0.5,
                format="%.1f",
            ),
            "sugerido_pct": st.column_config.NumberColumn("Sugerido (%)", format="%.1f"),
        },
    )

    if st.button("Guardar descuentos editados"):
        guardados, errores = 0, []
        for _, f in editado.iterrows():
            pid = f["id_producto"]
            nuevo = 0.0 if pd.isna(f["descuento_aplicado_pct"]) else float(f["descuento_aplicado_pct"])
            if abs(nuevo - float(original[pid])) < 0.01:
                continue
            try:
                punto_venta.guardar_descuento(C, pid, nuevo, sugerido_pct=float(sugerido[pid]))
                guardados += 1
            except ValueError as e:
                errores.append(str(e))
        for e in errores:
            st.error(e)
        if guardados:
            st.session_state.mensaje_desc = f"Descuentos guardados: {guardados} productos."
            st.session_state.version_desc += 1
            st.rerun()
        elif not errores:
            st.info("No hay cambios que guardar.")

    precios = punto_venta.tabla_precios(C)
    con = precios[precios["descuento_pct"] > 0]
    if len(con):
        st.subheader("Precios con descuento vigente")
        st.dataframe(
            con[["nombre", "precio_venta", "descuento_pct", "precio_final", "origen"]]
            .rename(columns={"descuento_pct": "descuento (%)", "precio_final": "precio final"}),
            hide_index=True, width="stretch",
        )


# ---------- Inventario ----------
def pagina_inventario():
    if not punto_venta.esta_inicializado():
        st.info("Carga primero un inventario con el botón de la barra lateral.")
        return

    if "version_inv" not in st.session_state:
        st.session_state.version_inv = 0
    if "mensaje_inv" in st.session_state:
        st.success(st.session_state.pop("mensaje_inv"))

    inv = punto_venta.cargar_inventario(C)

    st.subheader("Editar inventario")
    st.caption(
        "Cambia precios, existencias y datos directamente en la tabla. "
        "Para agregar un producto usa la última fila vacía. Para eliminar uno, "
        "selecciona la fila y pulsa Supr. Cada cambio queda en la bitácora al guardar."
    )
    editado = st.data_editor(
        inv,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"editor_inv_{st.session_state.version_inv}",
    )
    if st.button("Guardar cambios", type="primary"):
        try:
            n = punto_venta.guardar_cambios(editado, C)
            st.session_state.version_inv += 1
            st.session_state.mensaje_inv = (
                f"Cambios guardados. Movimientos registrados: {n}." if n else "No había cambios que guardar."
            )
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    st.subheader("Entrada de mercancía")
    nombres = dict(zip(inv["id_producto"], inv["nombre"]))
    with st.form("entrada_mercancia", clear_on_submit=True):
        pid = st.selectbox("Producto", list(nombres.keys()), format_func=lambda p: f"{nombres[p]} ({p})")
        cantidad = st.number_input("Cantidad que entra", min_value=1, value=1, step=1)
        costo = st.number_input("Costo unitario nuevo (0 = no cambiar)", min_value=0.0, value=0.0, step=0.5)
        if st.form_submit_button("Registrar entrada"):
            try:
                punto_venta.registrar_entrada(pid, int(cantidad), costo if costo > 0 else None, C)
                st.session_state.version_inv += 1
                st.session_state.mensaje_inv = f"Entrada registrada: +{int(cantidad)} de {nombres[pid]}."
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    with st.expander("Bitácora de movimientos"):
        mov = punto_venta.cargar_movimientos(C)
        if mov.empty:
            st.caption("Todavía no hay movimientos registrados.")
        else:
            st.dataframe(mov.iloc[::-1], hide_index=True, width="stretch")


if seccion == "Análisis":
    pagina_analisis()
elif seccion == "Descuentos":
    pagina_descuentos()
else:
    pagina_inventario()
