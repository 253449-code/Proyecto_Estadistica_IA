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
    /* Fondo general - Eliminado para respetar el Modo Oscuro nativo de Streamlit */
    /* .stApp { background-color: transparent; } */

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
        background: #262730; /* Gris oscuro nativo de Streamlit */
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); /* Sombra adaptada a oscuros */
        border-left: 4px solid #4da6ff; /* Borde azul brillante */
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff; /* Texto blanco para que se lea perfectamente */
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
st.markdown("""
<div class="section-card">
    <div class="section-title">🗂️ Módulo 1 · Datos Cargados</div>
</div>
""", unsafe_allow_html=True)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Observaciones (n)", f"{n:,}")
col_info2.metric("Media muestral (x̄)", f"{serie.mean():.4f}")
col_info3.metric("Desv. Est. Poblacional (σ)", f"{sigma:.4f}")
col_info4.metric("Error Estándar (σ/√n)", f"{sigma / np.sqrt(n):.4f}")

st.markdown("**Vista previa de los datos:**")
st.dataframe(serie.head(10).to_frame(), use_container_width=True, height=200)

# =============================================================================
# MÓDULO 2: VISUALIZACIÓN DE DISTRIBUCIONES
# =============================================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">📈 Módulo 2 · Visualización de Distribuciones</div>
</div>
""", unsafe_allow_html=True)

col_hist, col_box = st.columns(2)

# Paleta de colores consistente
COLOR_PRINCIPAL = "#0f3460"
COLOR_ACENTO    = "#e94560"

# --- Columna 1: Histograma con KDE ---
with col_hist:
    st.markdown("**Histograma + Curva KDE**")
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    fig_hist.patch.set_facecolor('#f7faff')
    ax_hist.set_facecolor('#f7faff')

    # sns.histplot dibuja el histograma y, con kde=True, superpone la curva de densidad
    sns.histplot(
        serie, kde=True, ax=ax_hist,
        color=COLOR_PRINCIPAL, alpha=0.5,
        edgecolor='white', linewidth=0.5
    )
    # Línea vertical para la media muestral
    ax_hist.axvline(serie.mean(), color=COLOR_ACENTO, linestyle='--', linewidth=2,
                    label=f'x̄ = {serie.mean():.2f}')
    ax_hist.set_xlabel("Valor", fontsize=10)
    ax_hist.set_ylabel("Frecuencia", fontsize=10)
    ax_hist.set_title("Distribución de la Muestra", fontsize=11, fontweight='bold', color=COLOR_PRINCIPAL)
    ax_hist.legend(fontsize=9)
    ax_hist.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

# --- Columna 2: Boxplot ---
with col_box:
    st.markdown("**Diagrama de Caja (Boxplot)**")
    fig_box, ax_box = plt.subplots(figsize=(6, 4))
    fig_box.patch.set_facecolor('#f7faff')
    ax_box.set_facecolor('#f7faff')

    sns.boxplot(
        y=serie, ax=ax_box,
        color=COLOR_PRINCIPAL, width=0.4,
        flierprops=dict(marker='o', color=COLOR_ACENTO, markersize=5, alpha=0.7)
    )
    ax_box.set_ylabel("Valor", fontsize=10)
    ax_box.set_title("Distribución y Outliers", fontsize=11, fontweight='bold', color=COLOR_PRINCIPAL)
    ax_box.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_box)
    plt.close(fig_box)

# --- Sección de análisis del estudiante (juicio cualitativo antes de ver resultados formales) ---
st.markdown("---")
st.markdown('<div class="reflection-header">✏️ Análisis del Estudiante — Responde antes de ver los resultados formales</div>', unsafe_allow_html=True)

col_preg1, col_preg2, col_preg3 = st.columns(3)

with col_preg1:
    resp_normal = st.radio(
        "¿La distribución parece normal?",
        ["Sí, parece normal", "No, no parece normal", "No estoy seguro/a"],
        index=2
    )
with col_preg2:
    resp_sesgo = st.radio(
        "¿Hay sesgo en la distribución?",
        ["Sin sesgo (simétrica)", "Sesgo a la derecha (cola derecha)", "Sesgo a la izquierda (cola izquierda)"],
        index=0
    )
with col_preg3:
    resp_outliers = st.radio(
        "¿Se observan outliers en el boxplot?",
        ["Sí, hay outliers visibles", "No, no se observan outliers", "Posiblemente, no estoy seguro/a"],
        index=1
    )



# =============================================================================
# MÓDULO 3: PRUEBA DE HIPÓTESIS Z
# =============================================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">📊 Módulo 3 · Prueba de Hipótesis</div>
</div>
""", unsafe_allow_html=True)

# --- Supuestos explícitos (pedagógicamente importante) ---
st.info(
    "**Supuestos de la Prueba Z aplicados aquí:**  \n"
    "① La varianza poblacional (σ²) es **conocida**.  \n"
    "② La muestra es **grande** (n ≥ 30), lo que, por el Teorema Central del Límite, "
    "garantiza que la distribución muestral de x̄ es aproximadamente normal."
)

# --- Parámetros de la prueba ---
col_h0, col_tipo, col_alpha = st.columns(3)

with col_h0:
    mu_0 = st.number_input(
        "Media hipotética H₀ (μ₀)",
        value=round(float(serie.mean()), 1),
        step=0.5,
        help="Este es el valor que se asume verdadero bajo la hipótesis nula."
    )

with col_tipo:
    tipo_prueba = st.selectbox(
        "Tipo de prueba:",
        ["Bilateral (H₁: μ ≠ μ₀)", "Cola izquierda (H₁: μ < μ₀)", "Cola derecha (H₁: μ > μ₀)"]
    )

with col_alpha:
    alpha = st.selectbox(
        "Nivel de significancia (α):",
        [0.01, 0.05, 0.10],
        index=1,
        help="Probabilidad máxima que se acepta de cometer un Error Tipo I (rechazar H₀ siendo verdadera)."
    )

st.markdown("---")

# =============================================================================
# CÁLCULOS DE LA PRUEBA Z
# =============================================================================
# Paso 1: Media muestral
x_barra = serie.mean()

# Paso 2: Error estándar de la media → σ / √n
# (El error estándar mide cuánto varía x̄ de muestra a muestra)
error_estandar = sigma / np.sqrt(n)

# Paso 3: Estadístico Z
# Fórmula: Z = (x̄ - μ₀) / (σ / √n)
# Nos dice a cuántas desviaciones estándar de la media hipotética está nuestra media muestral
Z_calc = (x_barra - mu_0) / error_estandar

# Paso 4: P-value y valor(es) crítico(s) según el tipo de prueba
# stats.norm.cdf(z) = P(Z ≤ z) bajo la distribución normal estándar N(0,1)
if "Bilateral" in tipo_prueba:
    # Prueba bilateral: la región de rechazo está en AMBAS colas
    # p-value = 2 × P(Z ≥ |Z_calc|)
    p_value = 2 * (1 - stats.norm.cdf(abs(Z_calc)))
    # Valor crítico: ±z_{α/2}  (se divide α entre 2 porque hay dos colas)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_criticos = [-z_crit, z_crit]
    # Se rechaza H₀ si |Z_calc| > z_crítico
    rechaza = abs(Z_calc) > z_crit

elif "izquierda" in tipo_prueba:
    # Cola izquierda: la región de rechazo está en el EXTREMO INFERIOR
    # p-value = P(Z ≤ Z_calc)
    p_value = stats.norm.cdf(Z_calc)
    z_crit = stats.norm.ppf(alpha)    # valor crítico negativo
    z_criticos = [z_crit]
    # Se rechaza H₀ si Z_calc < z_crítico
    rechaza = Z_calc < z_crit

else:  # Cola derecha
    # Cola derecha: la región de rechazo está en el EXTREMO SUPERIOR
    # p-value = P(Z ≥ Z_calc) = 1 - CDF(Z_calc)
    p_value = 1 - stats.norm.cdf(Z_calc)
    z_crit = stats.norm.ppf(1 - alpha)  # valor crítico positivo
    z_criticos = [z_crit]
    # Se rechaza H₀ si Z_calc > z_crítico
    rechaza = Z_calc > z_crit

# =============================================================================
# MOSTRAR RESULTADOS NUMÉRICOS
# =============================================================================
st.markdown("### 📐 Resultados de la Prueba")

# Fórmula explícita para el estudiante
st.markdown(f"""
<div class="formula-box">
Z = (x̄ − μ₀) / (σ / √n)
  = ({x_barra:.4f} − {mu_0}) / ({sigma:.4f} / √{n})
  = {x_barra - mu_0:.4f} / {error_estandar:.4f}
  = <strong>{Z_calc:.4f}</strong>
</div>
""", unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("x̄ (Media muestral)",  f"{x_barra:.4f}")
col_m2.metric("σ/√n (Error estándar)", f"{error_estandar:.4f}")
col_m3.metric("Z calculado",           f"{Z_calc:.4f}")
col_m4.metric("p-value",               f"{p_value:.5f}")
col_m5.metric("Z crítico(s)",
               " / ".join([f"{z:.4f}" for z in z_criticos]))

# --- Decisión estadística ---
st.markdown("### ⚖️ Decisión Estadística")
if rechaza:
    st.markdown(f"""
    <div class="decision-reject">
        🚫 Se RECHAZA H₀ (μ = {mu_0})<br>
        <small>p-value ({p_value:.5f}) &lt; α ({alpha}) — El estadístico Z cae en la región de rechazo.</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="decision-fail">
        ✅ No se rechaza H₀ (μ = {mu_0})<br>
        <small>p-value ({p_value:.5f}) ≥ α ({alpha}) — No hay evidencia suficiente para rechazar H₀.</small>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# GRÁFICA DE LA CAMPANA DE GAUSS — Región de rechazo y estadístico Z
# =============================================================================
st.markdown("### 🔔 Distribución Normal Estándar N(0,1) — Región de Rechazo")

fig_z, ax_z = plt.subplots(figsize=(10, 4.5))
fig_z.patch.set_facecolor('#f7faff')
ax_z.set_facecolor('#f7faff')

# Rango del eje x (cubrimos ±4 desviaciones estándar)
x_rango = np.linspace(-4.5, 4.5, 600)
y_normal = stats.norm.pdf(x_rango)   # f(x) de N(0,1)

# Curva de la distribución normal estándar
ax_z.plot(x_rango, y_normal, color=COLOR_PRINCIPAL, linewidth=2.5, zorder=3)
ax_z.fill_between(x_rango, y_normal, alpha=0.07, color=COLOR_PRINCIPAL)  # área bajo la curva (tenue)

# --- Sombreado de la(s) región(es) de rechazo ---
COLOR_RECHAZO = "#e94560"

if "Bilateral" in tipo_prueba:
    # Ambas colas: x ≤ -z_crit  y  x ≥ +z_crit
    mascara_izq = x_rango <= z_criticos[0]
    mascara_der = x_rango >= z_criticos[1]
    ax_z.fill_between(x_rango, y_normal, where=mascara_izq, color=COLOR_RECHAZO, alpha=0.45, label=f"Región de rechazo (α/2={alpha/2})")
    ax_z.fill_between(x_rango, y_normal, where=mascara_der, color=COLOR_RECHAZO, alpha=0.45)
    # Líneas verticales para los valores críticos
    for zc in z_criticos:
        ax_z.axvline(zc, color=COLOR_RECHAZO, linestyle='--', linewidth=1.5,
                     label=f"Z crítico = {zc:.3f}" if zc > 0 else None)
        ax_z.axvline(zc, color=COLOR_RECHAZO, linestyle='--', linewidth=1.5)

elif "izquierda" in tipo_prueba:
    mascara = x_rango <= z_criticos[0]
    ax_z.fill_between(x_rango, y_normal, where=mascara, color=COLOR_RECHAZO, alpha=0.45,
                      label=f"Región de rechazo (α={alpha})")
    ax_z.axvline(z_criticos[0], color=COLOR_RECHAZO, linestyle='--', linewidth=1.5,
                 label=f"Z crítico = {z_criticos[0]:.3f}")

else:  # Cola derecha
    mascara = x_rango >= z_criticos[0]
    ax_z.fill_between(x_rango, y_normal, where=mascara, color=COLOR_RECHAZO, alpha=0.45,
                      label=f"Región de rechazo (α={alpha})")
    ax_z.axvline(z_criticos[0], color=COLOR_RECHAZO, linestyle='--', linewidth=1.5,
                 label=f"Z crítico = {z_criticos[0]:.3f}")

# --- Línea/punto del estadístico Z calculado ---
COLOR_Z_CALC = "#f6ad55"
ax_z.axvline(Z_calc, color=COLOR_Z_CALC, linewidth=2.5, linestyle='-', zorder=5,
             label=f"Z calculado = {Z_calc:.3f}")
# Punto sobre la curva para el estadístico Z
y_z_calc = stats.norm.pdf(Z_calc)
ax_z.scatter([Z_calc], [y_z_calc], color=COLOR_Z_CALC, s=100, zorder=6, edgecolor='white', linewidth=1.5)

# Anotación del valor de Z calculado
ax_z.annotate(
    f"Z = {Z_calc:.3f}",
    xy=(Z_calc, y_z_calc),
    xytext=(Z_calc + 0.3, y_z_calc + 0.03),
    fontsize=9, color=COLOR_Z_CALC, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=COLOR_Z_CALC, lw=1.5)
)

ax_z.set_xlabel("Valores de Z", fontsize=11)
ax_z.set_ylabel("Densidad de probabilidad f(Z)", fontsize=11)
ax_z.set_title(f"Distribución Normal Estándar · Prueba Z · Tipo: {tipo_prueba.split('(')[0].strip()}", 
               fontsize=12, fontweight='bold', color=COLOR_PRINCIPAL)
ax_z.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax_z.spines[['top', 'right']].set_visible(False)
ax_z.set_xlim(-4.5, 4.5)
plt.tight_layout()
st.pyplot(fig_z)
plt.close(fig_z)



# =============================================================================
# MÓDULO 4: INTEGRACIÓN CON GOOGLE GEMINI
# =============================================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">🤖 Módulo 4 · Asistente IA — Google Gemini</div>
</div>
""", unsafe_allow_html=True)

# --- Construcción del prompt (NUNCA se envían datos crudos, solo el resumen estadístico) ---
tipo_prueba_corto = tipo_prueba.split("(")[0].strip()
decision_texto    = "Se rechaza H₀" if rechaza else "No se rechaza H₀"

prompt_ia = f"""
Eres un profesor universitario de estadística. Un estudiante realizó una Prueba Z con los siguientes parámetros:

• Media muestral (x̄): {x_barra:.4f}
• Media hipotética bajo H₀ (μ₀): {mu_0}
• Tamaño de muestra (n): {n}
• Desviación estándar poblacional conocida (σ): {sigma:.4f}
• Error estándar (σ/√n): {error_estandar:.4f}
• Nivel de significancia (α): {alpha}
• Tipo de prueba: {tipo_prueba_corto}
• Estadístico Z calculado: {Z_calc:.4f}
• Valor(es) crítico(s) Z: {', '.join([f'{z:.4f}' for z in z_criticos])}
• p-value: {p_value:.5f}
• Decisión: {decision_texto}

Por favor:
1. Explica paso a paso la lógica de la decisión tomada ({decision_texto}), incluyendo la comparación del p-value con α.
2. Interpreta qué significa este resultado en términos prácticos (sin acceso a los datos, usa los parámetros dados).
3. Evalúa si los supuestos de la Prueba Z (varianza conocida y n ≥ 30) son razonables dado el contexto.
4. Menciona qué error estadístico (Tipo I o Tipo II) sería relevante considerar en este caso y por qué.
5. Responde en español, con un tono didáctico y accesible para un estudiante universitario de estadística.
"""

col_boton, col_estado = st.columns([1, 3])

with col_boton:
    solicitar_ia = st.button("🧠 Solicitar análisis a la IA", use_container_width=True)

if solicitar_ia:
    # Intentamos obtener la API key desde el archivo .env
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error(
            "⚠️ **No se encontró la API key de Gemini.**  \n"
            "Crea un archivo `.env` en la misma carpeta que `app.py` con el contenido:  \n"
            "`GEMINI_API_KEY=tu_llave_aqui`"
        )
    else:
        try:
            # Importamos aquí para no fallar si la librería no está instalada
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            # CAMBIO A GEMINI-PRO PARA EVITAR EL ERROR 404 DE LA LIBRERÍA
          # USAMOS EL MODELO MÁS NUEVO Y ACTIVO
            modelo = genai.GenerativeModel("gemini-2.5-flash")

            with st.spinner("🔍 Gemini está analizando los resultados..."):
                respuesta = modelo.generate_content(prompt_ia)
                texto_respuesta = respuesta.text

            st.session_state["respuesta_ia"] = texto_respuesta

        except ImportError:
            st.error("⚠️ Instala la librería: `pip install google-generativeai`")
        except Exception as e:
            st.error(f"❌ Error al comunicarse con Gemini: {e}")

# Mostrar la respuesta si existe en session_state
if "respuesta_ia" in st.session_state:
    st.markdown("#### 💬 Análisis de Gemini:")
    st.info(st.session_state["respuesta_ia"])

st.markdown("---")

# Bloque de comparación directa
st.subheader("⚖️ Comparativa: Humano vs. IA")
col1, col2 = st.columns(2)

with col1:
    st.info(f"**Tu análisis visual:**\n- Normalidad: {resp_normal}\n- Sesgo: {resp_sesgo}\n- Outliers: {resp_outliers}")

with col2:
    st.success(f"**Análisis de la IA:**\nConsulta el texto superior para ver la validación de tus hipótesis y la interpretación de la Prueba Z.")
    
# =============================================================================
# SECCIÓN DE REFLEXIÓN CRÍTICA DEL ESTUDIANTE (núcleo pedagógico)
# =============================================================================
st.markdown("#### ✍️ Tu Reflexión — Comparación con la IA")
st.markdown(
    "Este es el momento más importante del ejercicio. Compara el análisis de la IA con tu propio "
    "razonamiento estadístico. ¿Coincidís? ¿Detectas errores, imprecisiones o 'alucinaciones'?"
)

reflexion_estudiante = st.text_area(
    "Escribe tus conclusiones aquí:",
    placeholder=(
        "Ejemplo:\n"
        "- La IA concluyó correctamente que [...]  \n"
        "- Sin embargo, creo que la IA se equivocó en [...] porque [...]  \n"
        "- El p-value de {:.5f} indica que [...]  \n"
        "- El supuesto de σ conocida es razonable/no razonable porque [...]  \n"
        "- Un Error Tipo I en este contexto significaría [...]".format(p_value)
    ),
    height=220,
    key="reflexion"
)

if reflexion_estudiante:
    st.success("✅ Reflexión guardada. ¡Excelente trabajo crítico!")
    # Resumen de los parámetros elegidos por el estudiante para el módulo 2
    st.markdown("**Resumen de tu análisis visual (Módulo 2):**")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.info(f"**Normalidad:** {resp_normal}")
    col_r2.info(f"**Sesgo:** {resp_sesgo}")
    col_r3.info(f"**Outliers:** {resp_outliers}")

# =============================================================================
# PIE DE PÁGINA
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.82rem; padding: 0.5rem;'>"
    "Laboratorio de Prueba Z · Estadística Inferencial · Herramienta Educativa · "
    "Los datos nunca se envían a servicios externos — solo se transmite el resumen estadístico."
    "</div>",
    unsafe_allow_html=True
)