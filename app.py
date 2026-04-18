import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        
    else:
        st.error("El archivo no contiene columnas numéricas válidas para el análisis.")
else:
    st.info("👈 Por favor, genera datos sintéticos o sube un archivo CSV en el menú lateral para comenzar.")