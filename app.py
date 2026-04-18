import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Configuración inicial de la página
st.set_page_config(page_title="Prueba de Hipótesis y Análisis", layout="wide")
st.title("📊 Análisis Estadístico y Prueba de Hipótesis")
st.markdown("Aplicación interactiva para evaluar distribuciones y realizar pruebas Z.")

# --- MÓDULO 1: CARGA DE DATOS ---
st.sidebar.header("1. Carga de Datos")
tipo_datos = st.sidebar.radio("Elige la fuente de datos:", ["Generar datos sintéticos", "Subir archivo CSV"])

df = None
columna_seleccionada = None

if tipo_datos == "Generar datos sintéticos":
    st.sidebar.subheader("Parámetros de datos sintéticos")
    n_muestras = st.sidebar.slider("Tamaño de muestra (n >= 30)", min_value=30, max_value=1000, value=100)
    media_real = st.sidebar.number_input("Media real", value=50.0)
    desviacion_real = st.sidebar.number_input("Desviación estándar", value=10.0)
    
    if st.sidebar.button("Generar Datos"):
        # Generamos una distribución normal aleatoria
        datos = np.random.normal(loc=media_real, scale=desviacion_real, size=n_muestras)
        df = pd.DataFrame({"Variable_Sintetica": datos})
        st.success(f"Se generaron {n_muestras} datos sintéticos.")

elif tipo_datos == "Subir archivo CSV":
    archivo_subido = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo_subido is not None:
        df = pd.read_csv(archivo_subido)
        st.success("Archivo cargado correctamente.")

        df = st.session_state.get('df_proyecto', None)

# Si ya tenemos datos (ya sea CSV o sintéticos), procedemos
if df is not None:
    st.subheader("Vista previa de los datos")
    st.write(df.head())
    
    # Selección de variable
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    if columnas_numericas:
        columna_seleccionada = st.selectbox("Selecciona la variable a analizar:", columnas_numericas)
        datos_analisis = df[columna_seleccionada].dropna()
        
        # --- MÓDULO 2: VISUALIZACIÓN ---
        st.markdown("---")
        st.header("📈 Visualización de Distribuciones")
        
        # Crear columnas para mostrar gráficos lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Histograma y KDE")
            fig, ax = plt.subplots()
            sns.histplot(datos_analisis, kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Distribución de {columna_seleccionada}")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Boxplot (Diagrama de Caja)")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=datos_analisis, ax=ax2, color="lightgreen")
            ax2.set_title(f"Boxplot de {columna_seleccionada}")
            st.pyplot(fig2)
            
        # Preguntas interactivas requeridas
        st.markdown("### 📝 Análisis del estudiante")
        st.radio("¿La distribución parece normal?", ["Sí, tiene forma de campana", "No, está sesgada o es irregular"])
        st.radio("¿Hay sesgo evidente?", ["Sin sesgo (Simétrica)", "Sesgo a la derecha (Positivo)", "Sesgo a la izquierda (Negativo)"])
        st.radio("¿Se observan valores atípicos (outliers) en el boxplot?", ["Sí", "No"])

        # --- MÓDULO 3: PRUEBA DE HIPÓTESIS (PRUEBA Z) ---
        st.markdown("---")
        st.header("⚖️ Módulo 3: Prueba de Hipótesis (Prueba Z)")
        st.info("Nota: Se asume varianza poblacional conocida y una muestra grande (n ≥ 30).")

        col_input1, col_input2 = st.columns(2)
        with col_input1:
            mu_0 = st.number_input("Hipótesis Nula H₀ (Media hipotética):", value=50.0)
            sigma_pob = st.number_input("Desviación estándar poblacional (σ):", value=10.0, min_value=0.01)
        with col_input2:
            tipo_prueba = st.selectbox("Tipo de prueba (Hipótesis Alternativa H₁):", 
                                       ["Bilateral (≠)", "Cola izquierda (<)", "Cola derecha (>)"])
            alpha = st.selectbox("Nivel de significancia (α):", [0.01, 0.05, 0.10], index=1)

        # Botón para ejecutar la prueba
        if st.button("Ejecutar Prueba Z"):
            # 1. Cálculos Estadísticos
            n = len(datos_analisis)
            media_muestral = datos_analisis.mean()
            
            if n < 30:
                st.warning(f"Cuidado: El tamaño de muestra es {n}. Se recomienda n ≥ 30 para la Prueba Z.")

            error_estandar = sigma_pob / np.sqrt(n)
            z_calc = (media_muestral - mu_0) / error_estandar

            # 2. Determinar Regiones Críticas y p-value
            if tipo_prueba == "Bilateral (≠)":
                z_critico = norm.ppf(1 - alpha/2)
                p_value = 2 * (1 - norm.cdf(abs(z_calc)))
                rechazar_h0 = abs(z_calc) > z_critico
            elif tipo_prueba == "Cola izquierda (<)":
                z_critico = norm.ppf(alpha)
                p_value = norm.cdf(z_calc)
                rechazar_h0 = z_calc < z_critico
            else: # Cola derecha (>)
                z_critico = norm.ppf(1 - alpha)
                p_value = 1 - norm.cdf(z_calc)
                rechazar_h0 = z_calc > z_critico

            # 3. Mostrar Resultados
            st.subheader("Resultados de la Prueba")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            res_col1.metric("Media Muestral (x̄)", f"{media_muestral:.2f}")
            res_col2.metric("Tamaño (n)", n)
            res_col3.metric("Estadístico Z", f"{z_calc:.4f}")
            res_col4.metric("Valor p (p-value)", f"{p_value:.4f}")

            if rechazar_h0:
                st.error(f"**Decisión:** Se RECHAZA la Hipótesis Nula H₀. (p-value ≤ α)")
            else:
                st.success(f"**Decisión:** NO SE RECHAZA la Hipótesis Nula H₀. (p-value > α)")

            # 4. Gráfica de la Campana de Gauss con Zona de Rechazo
            st.markdown("#### Visualización de la Región de Rechazo")
            fig_z, ax_z = plt.subplots(figsize=(8, 4))
            
            # Eje X para la campana N(0,1)
            x_val = np.linspace(-4, 4, 1000)
            y_val = norm.pdf(x_val, 0, 1)
            ax_z.plot(x_val, y_val, color='black', linewidth=1.5)
            
            # Sombrear región de rechazo
            if tipo_prueba == "Bilateral (≠)":
                ax_z.fill_between(x_val, y_val, where=(x_val < -z_critico), color='red', alpha=0.3, label='Rechazo')
                ax_z.fill_between(x_val, y_val, where=(x_val > z_critico), color='red', alpha=0.3)
                ax_z.axvline(-z_critico, color='red', linestyle='--', label=f'Z Crítico (±{z_critico:.2f})')
                ax_z.axvline(z_critico, color='red', linestyle='--')
            elif tipo_prueba == "Cola izquierda (<)":
                ax_z.fill_between(x_val, y_val, where=(x_val < z_critico), color='red', alpha=0.3, label='Rechazo')
                ax_z.axvline(z_critico, color='red', linestyle='--', label=f'Z Crítico ({z_critico:.2f})')
            else: # Cola derecha
                ax_z.fill_between(x_val, y_val, where=(x_val > z_critico), color='red', alpha=0.3, label='Rechazo')
                ax_z.axvline(z_critico, color='red', linestyle='--', label=f'Z Crítico ({z_critico:.2f})')

            # Marcar el Z calculado
            ax_z.axvline(z_calc, color='blue', linewidth=2, label=f'Z Calculado ({z_calc:.2f})')
            
            ax_z.set_title("Distribución Normal Estándar Z ~ N(0,1)")
            ax_z.set_xlabel("Valores Z")
            ax_z.set_ylabel("Densidad de Probabilidad")
            ax_z.legend()
            st.pyplot(fig_z)

            # Guardar datos en session_state para el futuro Módulo 4 (IA)
            st.session_state['datos_ia'] = {
                'mu_0': mu_0, 'sigma': sigma_pob, 'alpha': alpha, 'n': n,
                'media_muestral': media_muestral, 'z_calc': z_calc, 'p_value': p_value,
                'tipo_prueba': tipo_prueba, 'decision': "Rechazar H0" if rechazar_h0 else "No Rechazar H0"
            }
        
    else:
        st.error("El archivo no contiene columnas numéricas válidas para el análisis.")
else:
    st.info("👈 Por favor, genera datos sintéticos o sube un archivo CSV en el menú lateral para comenzar.")