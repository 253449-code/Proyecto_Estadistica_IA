# =============================================================================
# app.py - Laboratorio Interactivo de Prueba Z con Asistente IA
# =============================================================================
# Descripción: Aplicación educativa para aprender pruebas de hipótesis Z,
# visualización de distribuciones y comparación con análisis de IA (Google Gemini).
#
# Librerías requeridas:
#   pip install streamlit pandas numpy scipy matplotlib seaborn google-generativeai python-dotenv
# =============================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (donde el estudiante guarda GEMINI_API_KEY)
load_dotenv()

# =============================================================================
# CONFIGURACIÓN GLOBAL DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Laboratorio de Prueba Z",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS personalizados para una interfaz más limpia y académica ---
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #F0F4F8; }

    /* Encabezado principal */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 0.95rem; opacity: 0.75; margin: 0.4rem 0 0; }

    /* Tarjetas de sección */
    .section-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #0f3460;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f3460;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Decisión estadística */
    .decision-reject {
        background: #fff0f0; border: 2px solid #e53e3e;
        border-radius: 8px; padding: 1rem 1.5rem;
        color: #c53030; font-weight: 700; font-size: 1.1rem;
        text-align: center;
    }
    .decision-fail {
        background: #f0fff4; border: 2px solid #38a169;
        border-radius: 8px; padding: 1rem 1.5rem;
        color: #276749; font-weight: 700; font-size: 1.1rem;
        text-align: center;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #c0c0d0 !important; }

    /* Fórmula matemática destacada */
    .formula-box {
        background: #f7faff;
        border: 1px solid #bee3f8;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        color: #2c5282;
        margin: 0.5rem 0;
    }

    /* Área de reflexión estudiantil */
    .reflection-header {
        background: linear-gradient(90deg, #f6ad55, #ed8936);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ENCABEZADO PRINCIPAL
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>📊 Laboratorio Interactivo de Prueba Z</h1>
    <p>Estadística Inferencial · Prueba de Hipótesis con Asistente IA · Herramienta Educativa</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR: MÓDULO 1 — CARGA Y CONFIGURACIÓN DE DATOS
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuración de Datos")
    st.markdown("---")

    # --- Elección de la fuente de datos ---
    fuente = st.radio(
        "Fuente de datos:",
        ["🎲 Generar datos sintéticos", "📂 Subir archivo CSV"],
        index=0
    )

    datos = None           # DataFrame final (columna seleccionada como Serie)
    sigma_poblacional = None  # Desviación estándar poblacional (σ)

    # ---- OPCIÓN A: Datos sintéticos ----------------------------------------
    if fuente == "🎲 Generar datos sintéticos":
        st.markdown("### Parámetros Poblacionales")

        mu_pob = st.number_input(
            "Media poblacional (μ)", value=100.0, step=1.0,
            help="La media real de la población de la que se extrae la muestra."
        )
        sigma_pob = st.number_input(
            "Desv. estándar poblacional (σ)", value=15.0, min_value=0.01, step=0.5,
            help="La desviación estándar CONOCIDA de la población (supuesto de la Prueba Z)."
        )
        n_muestra = st.slider(
            "Tamaño de muestra (n)", min_value=30, max_value=2000,
            value=100, step=10,
            help="Mínimo 30 para cumplir el supuesto de muestra grande del Teorema Central del Límite."
        )

        # Semilla para reproducibilidad (buena práctica académica)
        semilla = st.number_input("Semilla aleatoria", value=42, step=1)

        if st.button("🔄 Generar muestra", use_container_width=True):
            # Generamos la muestra con distribución normal usando los parámetros del usuario
            np.random.seed(int(semilla))
            muestra = np.random.normal(loc=mu_pob, scale=sigma_pob, size=n_muestra)
            # Guardamos en session_state para mantener persistencia entre reruns
            st.session_state["datos"] = pd.Series(muestra, name="Muestra")
            st.session_state["sigma"] = sigma_pob
            st.session_state["n"] = n_muestra
            st.success(f"✅ Muestra generada: n={n_muestra}")

    # ---- OPCIÓN B: Archivo CSV ---------------------------------------------
    else:
        st.markdown("### Cargar CSV")
        archivo = st.file_uploader("Selecciona un archivo .csv", type=["csv"])

        if archivo is not None:
            try:
                df_csv = pd.read_csv(archivo)
                # Solo permitimos columnas numéricas (requisito del módulo)
                cols_numericas = df_csv.select_dtypes(include=np.number).columns.tolist()

                if not cols_numericas:
                    st.error("⚠️ El CSV no contiene columnas numéricas.")
                else:
                    col_sel = st.selectbox("Selecciona la columna a analizar:", cols_numericas)
                    muestra_csv = df_csv[col_sel].dropna()

                    # Verificar que n >= 30 para poder usar la Prueba Z
                    if len(muestra_csv) < 30:
                        st.warning(f"⚠️ n={len(muestra_csv)}. Se necesitan al menos 30 observaciones para la Prueba Z.")
                    else:
                        sigma_ingresada = st.number_input(
                            "Desv. estándar poblacional σ (conocida)", value=float(muestra_csv.std()),
                            min_value=0.01, step=0.1,
                            help="En la Prueba Z, σ debe ser CONOCIDA. Si no la conoces, este valor es una aproximación."
                        )
                        if st.button("✅ Usar estos datos", use_container_width=True):
                            st.session_state["datos"] = muestra_csv.reset_index(drop=True)
                            st.session_state["sigma"] = sigma_ingresada
                            st.session_state["n"] = len(muestra_csv)
                            st.success(f"Datos cargados: n={len(muestra_csv)}")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    st.markdown("---")
    st.caption("📘 Aplicación educativa · Prueba Z · © 2024")


# =============================================================================
# VERIFICAR SI YA HAY DATOS CARGADOS EN SESSION STATE
# =============================================================================
# Usamos st.session_state para conservar los datos entre interacciones de Streamlit

if "datos" not in st.session_state:
    # No hay datos aún → instruimos al usuario
    st.info("👈 **Comienza en el panel izquierdo:** genera datos sintéticos o sube un CSV, luego haz clic en el botón correspondiente.")
    st.stop()  # Detenemos la ejecución; el resto no se muestra sin datos

# Recuperamos los datos desde session_state
serie = st.session_state["datos"]
sigma = st.session_state["sigma"]
n = st.session_state["n"]

# =============================================================================
# MÓDULO 1 (continuación): PREVISUALIZACIÓN DE DATOS
# =============================================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🗂️ Módulo 1 · Datos Cargados</div>', unsafe_allow_html=True)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Observaciones (n)", f"{n:,}")
col_info2.metric("Media muestral (x̄)", f"{serie.mean():.4f}")
col_info3.metric("Desv. Est. Poblacional (σ)", f"{sigma:.4f}")
col_info4.metric("Error Estándar (σ/√n)", f"{sigma / np.sqrt(n):.4f}")

st.markdown("**Vista previa de los datos:**")
st.dataframe(serie.head(10).to_frame(), use_container_width=True, height=200)
st.markdown('</div>', unsafe_allow_html=True)


