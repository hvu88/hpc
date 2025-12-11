import streamlit as st
import subprocess
import pandas as pd
import plotly.graph_objects as go
import re
import numpy as np
import os
import time
import shutil
import tempfile

# --- Configuración de la Página ---
st.set_page_config(page_title="Escalabidad - HPC", layout="wide")

st.title("🔥 Ecuación de calor 2D: Análisis de Escalabilidad")
st.markdown("""
Este dashboard permite visualizar el comportamiento del algoritmo ante diferentes cargas de trabajo ($N$) 
y evalúa simultáneamente la escalabilidad Fuerte y Débil.
""")

# --- Inicialización de Estado ---
if 'df_final' not in st.session_state:
    st.session_state.df_final = None

# --- Barra Lateral (Inputs) ---
st.sidebar.header("Configuración del Experimento")

FILENAME_DEFAULT = "final_1000x1000.bin" 
FILENAME_BENCHMARK = "benchmark_final.bin" 

# 1. INPUT DE MÚLTIPLES MALLAS
mallas_input = st.sidebar.text_input("Tamaños de Malla (separar por comas)", value="100, 300, 500")

# 2. CONFIGURACIÓN DE FUENTE DE CALOR
heat_source = st.sidebar.selectbox(
    "Fuente de Calor",
    options=[
        "Bordes (Original)", 
        "Centro (Círculo r= 5% del tamaño de la malla)",
        "Centro (Franja Horizontal)"],
    index=0 
)
# Mapeo de IDs para C:
if heat_source == "Bordes (Original)":
    HEAT_SOURCE_ID = 0
elif heat_source == "Centro (Círculo r= 5% del tamaño de la malla)":
    HEAT_SOURCE_ID = 1
else:
    HEAT_SOURCE_ID = 2

# 3. ITERACIONES Y PROCESADORES
steps = st.sidebar.number_input("Iteraciones de Tiempo", min_value=1000, max_value=1000000, value=20000, step=5000)
max_procs = st.sidebar.slider("Máximo de Procesos", min_value=1, max_value=4, value=4)

# Parseo de inputs
try:
    N_VALUES = [int(x.strip()) for x in mallas_input.split(',')]
    N_VALUES.sort()
except:
    st.sidebar.error("Formato incorrecto. Usando default: 100, 300, 500")
    N_VALUES = [100, 300, 500]

st.sidebar.markdown("---")

# === FUNCIÓN HELPER PARA EJECUTAR MPI ===
def run_simulation(n, p, steps, heat_id):
    cmd = [
        "mpirun", "--allow-run-as-root", "-np", str(p), 
        "./simulacion_app", str(n), str(n), str(steps), str(heat_id)
    ]
    try:
        start_t = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_t = time.time()
        
        output = result.stdout
        match = re.search(r"Tiempo Final \(2D MPI\): ([0-9]+\.[0-9]+)", output)
        if match:
            return float(match.group(1)), True
        else:
            return end_t - start_t, False # Fallback si falla regex
    except Exception as e:
        return 0.0, False

# === BOTÓN DE EJECUCIÓN ===
if st.sidebar.button("🚀 Ejecutar Análisis"):
    
    # Limpieza inicial
    if os.path.exists(FILENAME_BENCHMARK): os.remove(FILENAME_BENCHMARK)
    if os.path.exists(FILENAME_DEFAULT): os.remove(FILENAME_DEFAULT)
    
    resultados = []
    
    # Variables para rastrear el archivo más grande
    max_n_simulated = 0 
    
    # Cálculo de progreso
    total_ops = (len(N_VALUES) * max_procs) + max_procs
    current_op = 0
    progress_bar = st.progress(0)
    
    st.divider()
    st.subheader("📝 Bitácora de Ejecución")
    log_container = st.container()

    # --- 1. EJECUCIÓN ESCALADO FUERTE ---
    for n_base in N_VALUES:
        for p in range(1, max_procs + 1):
            current_op += 1
            
            # Ejecutar
            t_val, _ = run_simulation(n_base, p, steps, HEAT_SOURCE_ID)
            
            # Lógica de guardado inteligente
            if n_base >= max_n_simulated:
                max_n_simulated = n_base
                if os.path.exists(FILENAME_DEFAULT):
                    shutil.copy(FILENAME_DEFAULT, FILENAME_BENCHMARK) 

            # Log
            with log_container:
                st.write(f"🔹 **Fuerte (N={n_base}, P={p})**: {t_val:.4f} s")
            
            resultados.append({
                "Categoria": "Fuerte", 
                "Etiqueta": f"N={n_base}",
                "N_Real": n_base, 
                "P": p, 
                "Tiempo": t_val
            })
            progress_bar.progress(current_op / total_ops)

    # --- 2. EJECUCIÓN ESCALADO DÉBIL ---
    n_weak_base = N_VALUES[0]
    
    for p in range(1, max_procs + 1):
        current_op += 1
        n_weak_scaled = int(n_weak_base * np.sqrt(p))
        
        # Ejecutar
        t_val, _ = run_simulation(n_weak_scaled, p, steps, HEAT_SOURCE_ID)
        
        # Lógica de guardado inteligente
        if n_weak_scaled >= max_n_simulated:
            max_n_simulated = n_weak_scaled
            if os.path.exists(FILENAME_DEFAULT):
                shutil.copy(FILENAME_DEFAULT, FILENAME_BENCHMARK)

        # Log
        with log_container:
            st.write(f"🔸 **Débil (P={p}, N={n_weak_scaled})**: {t_val:.4f} s")
        
        resultados.append({
            "Categoria": "Débil", 
            "Etiqueta": "Escalado débil (Trend)", 
            "N_Real": n_weak_scaled, 
            "P": p, 
            "Tiempo": t_val
        })
        progress_bar.progress(current_op / total_ops)

    # Procesamiento de datos
    if len(resultados) > 0:
        df = pd.DataFrame(resultados)
        df_final = pd.DataFrame()
        
        for etiqueta in df["Etiqueta"].unique():
            subset = df[df["Etiqueta"] == etiqueta].copy().sort_values("P")
            if not subset.empty:
                t_ref = subset.iloc[0]["Tiempo"]
                
                # Por defecto cálculo Fuerte
                subset["Speedup"] = t_ref / subset["Tiempo"]
                subset["Eficiencia"] = subset["Speedup"] / subset["P"]
                
                # 🛠️ CORRECCIÓN 1: Detección correcta de la etiqueta en español
                if "Escalado débil" in etiqueta:
                     subset["Eficiencia"] = t_ref / subset["Tiempo"]
                     subset["Speedup"] = subset["Eficiencia"] * subset["P"]

                df_final = pd.concat([df_final, subset])
        
        st.session_state.df_final = df_final
        st.rerun()

# === VISUALIZACIÓN DE RESULTADOS ===
if st.session_state.df_final is not None:
    df = st.session_state.df_final
    
    st.divider()
    st.header("📊 Resultados del Análisis")

    # ---------------------------------------------------------
    # 1. TABLA DE RESULTADOS
    # ---------------------------------------------------------
    st.subheader("📋 Tabla de Datos")
    st.dataframe(
        df.style.format({
            "Tiempo": "{:.4f}", 
            "Speedup": "{:.2f}x", 
            "Eficiencia": "{:.1%}"
        }), 
        use_container_width=True
    )

    # ---------------------------------------------------------
    # 2. GRÁFICAS
    # ---------------------------------------------------------
    st.subheader("📈 Comparativa Gráfica")

    col1, col2, col3 = st.columns(3)
    
    weak_label = "Escalado débil (Trend)"
    
    # 🛠️ CORRECCIÓN 2: Filtrar excluyendo explícitamente la etiqueta completa en español
    # Esto evita que se grafique como línea sólida en el bucle principal
    unique_labels = [l for l in df["Etiqueta"].unique() if l != weak_label]
    

   # Gráfico Tiempo
    fig_time = go.Figure()
    
    # 1. Bucle para las líneas de N (Escalado Fuerte)
    for label in unique_labels:
        data = df[df["Etiqueta"] == label]
        fig_time.add_trace(go.Scatter(
            x=data["P"], 
            y=data["Tiempo"],
            customdata=data["N_Real"], 
            mode='lines+markers', 
            name=label,
            # 🛠️ AJUSTE: Formato de dos decimales (.2f) y agregamos 's' de segundos
            hovertemplate="P: %{x}<br>N:%{customdata}<br>Tiempo: %{y:.2f} s<extra></extra>"
        ))
        
    # 2. Línea de Escalado Débil (Punteada)
    if weak_label in df["Etiqueta"].values:
        data_w = df[df["Etiqueta"] == weak_label]
        fig_time.add_trace(go.Scatter(
            x=data_w["P"], 
            y=data_w["Tiempo"],
            # 🛠️ NUEVO: Pasamos la columna 'N_Real' como datos personalizados
            customdata=data_w["N_Real"],
            mode='lines+markers', 
            name="Escalado débil (Ideal: Plano)", 
            line=dict(dash='dash', color='gray'),
            # 🛠️ AJUSTE: Leemos el dato personalizado usando %{customdata}
            hovertemplate="P: %{x}<br>N: %{customdata}<br>Tiempo: %{y:.2f} s<extra></extra>"
        ))
    
    fig_time.update_layout(
        title="1. Tiempo de Ejecución (s)", 
        xaxis_title="Procesos", 
        yaxis_title="Segundos", 
        legend=dict(orientation="h", y=1.1)
    )
    col1.plotly_chart(fig_time, use_container_width=True)
    
# Gráfico Speedup
    fig_speed = go.Figure()
    for label in unique_labels:
        data = df[df["Etiqueta"] == label]
        fig_speed.add_trace(go.Scatter(
            x=data["P"], 
            y=data["Speedup"],
            customdata=data["N_Real"], 
            mode='lines+markers', 
            name=label,
            # 🛠️ AJUSTE: Formato de dos decimales en el tooltip
            hovertemplate="P: %{x}<br>N:%{customdata}<br>Tiempo: %{y:.2f} s<extra></extra>"
        ))
        
    fig_speed.add_trace(go.Scatter(
        x=[1, max_procs], 
        y=[1, max_procs], 
        mode='lines', 
        name='Ideal', 
        line=dict(dash='dash', color='white'),
        hoverinfo='skip' # Opcional: para que la línea ideal no moleste al pasar el mouse
    ))
    
    fig_speed.update_layout(
        title="2. Speedup", 
        xaxis_title="Procesos", 
        yaxis_title="Factor (x)", 
        legend=dict(orientation="h", y=1.1)
    )
    col2.plotly_chart(fig_speed, use_container_width=True)

 # Gráfico Eficiencia
    fig_eff = go.Figure()
    
    # 1. Líneas de Escalado Fuerte (N fijo)
    for label in unique_labels:
        data = df[df["Etiqueta"] == label]
        fig_eff.add_trace(go.Scatter(
            x=data["P"], 
            y=data["Eficiencia"],
            customdata=data["N_Real"], 
            mode='lines+markers', 
            name=label,
            # 🛠️ AJUSTE: Formato de dos decimales
            hovertemplate="P: %{x}<br>N:%{customdata}<br>Tiempo: %{y:.2f} s<extra></extra>"
        ))
    
    # 2. Línea de Escalado Débil (Variable)
    if weak_label in df["Etiqueta"].values:
        data_w = df[df["Etiqueta"] == weak_label]
        fig_eff.add_trace(go.Scatter(
            x=data_w["P"], 
            y=data_w["Eficiencia"], 
            # 🛠️ EXTRA: Incluimos el N real en el tooltip
            customdata=data_w["N_Real"],
            mode='lines+markers', 
            name="Escalado débil (Consistencia)", 
            line=dict(dash='dash', color='gray'),
            # 🛠️ AJUSTE: Formato con N y dos decimales
            hovertemplate="P: %{x}<br>N: %{customdata}<br>Eficiencia: %{y:.2f}<extra></extra>"
        ))
    
    fig_eff.update_layout(
        title="3. Eficiencia", 
        xaxis_title="Procesos", 
        yaxis_title="Eficiencia (0-1)", 
        legend=dict(orientation="h", y=1.1)
    )
    col3.plotly_chart(fig_eff, use_container_width=True)
    
    # ---------------------------------------------------------
    # 3. DIAGNÓSTICO (MEJORADO: BASADO EN VARIACIÓN)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("🧠 Diagnóstico de Escalabilidad")
    
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.info("**Diagnóstico: Escalabilidad Fuerte**")
        largest_n_label = f"N={N_VALUES[-1]}"
        run_strong = df[(df["Etiqueta"] == largest_n_label)].sort_values("P")
        
        if not run_strong.empty:
            # Métricas extremas
            eff_start = run_strong.iloc[0]["Eficiencia"] # Debería ser 1.0
            eff_end = run_strong.iloc[-1]["Eficiencia"]
            p_start = run_strong.iloc[0]["P"]
            p_end = run_strong.iloc[-1]["P"]
            
            # CÁLCULO DE VARIACIÓN (Pérdida por núcleo añadido)
            # Cuánto perdemos cada vez que agregamos un core
            if p_end > p_start:
                loss_per_core = (eff_start - eff_end) / (p_end - p_start)
            else:
                loss_per_core = 0.0

            st.write(f"Análisis sobre malla **N={N_VALUES[-1]}**:")
            
            c1, c2 = st.columns(2)
            c1.metric("Eficiencia Final", f"{eff_end:.1%}")
            # Mostramos la variación. Si es negativa es ganancia (superlineal), positiva es pérdida.
            c2.metric("Degradación/Core", f"{-1*loss_per_core:.1}", help="Variación promedio de puntos porcentuales por proceso.")
            
            # Evaluación basada en la TASA DE CAMBIO, no solo en el final
            if loss_per_core < 0:
                st.success("🚀 **Super-Escalable:** Eficiencia mejor que el teórico ideal debido a la optimización de la memoria caché.")
            elif loss_per_core < 0.05: # Pierde menos de 5PP por core
                st.success("✅ **Escalabilidad Robusta:** La degradación de la curva de eficiencia es mínima. Soporta agregar más procesos")
            elif loss_per_core < 0.10: # Pierde entre 5PP y 10PP por core
                st.warning("⚠️ **Escalabilidad Sensible:** La curva de eficiencia se degrada visiblemente. Revisar comunicaciones.")
            else:
                st.error("❌ **No Escalable:** Pierde más de 0.1PP de eficiencia por cada proceso nuevo.")

    with d_col2:
        st.info("**Diagnóstico: Escalabilidad Débil**")
        run_weak = df[(df["Etiqueta"] == weak_label)].sort_values("P")
        
        if not run_weak.empty:
            # 1. Obtener Eficiencias inicial y final
            eff_start = run_weak.iloc[0]["Eficiencia"] # Debería ser cercano a 1.0
            eff_end = run_weak.iloc[-1]["Eficiencia"]
            
            # 2. Calcular rango de procesadores
            p_span = run_weak.iloc[-1]["P"] - run_weak.iloc[0]["P"]
            
            # 3. Calcular Degradación Promedio por Paso (Core añadido)
            # Cuánto cae la eficiencia (consistencia temporal) cada vez que añadimos un core
            if p_span > 0:
                loss_per_core_weak = (eff_start - eff_end) / p_span
            else: 
                loss_per_core_weak = 0.0

            st.write(f"Proyección desde **N={N_VALUES[0]}**:")
            
            c1, c2 = st.columns(2)
            c1.metric("Consistencia Final", f"{eff_end:.1%}")
            # Muestra cuánto se degrada la consistencia por cada core extra
            c2.metric("Degradación/Core", f"{-1*loss_per_core_weak:.1}", delta_color="inverse", help="Variación promedio de puntos porcentuales por proceso.")
            
            # Evaluación basada en la TASA DE DEGRADACIÓN
            if loss_per_core_weak < 0:
                st.success("🚀 **Ideal:** Eficiencia mejor que el teórico ideal debido a la optimización de la memoria caché.")
            elif loss_per_core_weak < 0.02: # Pierde menos de 2PP por core
                st.success("✅ **Ideal:** La eficiencia se mantiene casi constante.")
            elif loss_per_core_weak < 0.08: # Pierde entre 2PP y 8PP
                st.warning("⚠️ **Aceptable:** Ligera caída de la eficiencia al escalar.")
            else:
                st.error("❌ **Degradación:** La eficiencia no se mantiene constante.")

    # ---------------------------------------------------------
    # 4. MAPA DE CALOR ESTÁTICO (ROBUSTO)
    # ---------------------------------------------------------
    st.divider()
    
    n_detected = 0
    grid_final = None
    
    if os.path.exists(FILENAME_BENCHMARK):
        try:
            # 1. Detectar N real basado en el peso del archivo
            file_stats = os.stat(FILENAME_BENCHMARK)
            file_bytes = file_stats.st_size
            num_doubles = file_bytes // 8
            n_detected = int(np.sqrt(num_doubles))
            
            st.subheader(f"🌡️ Visualización (N={n_detected})")
            st.info(f"Se usa el valor de N mas grande detectado. **N={n_detected}** basado en la Tabla de Datos.")

            if n_detected * n_detected == num_doubles:
                data_raw = np.fromfile(FILENAME_BENCHMARK, dtype=np.float64)
                grid_final = data_raw.reshape((n_detected, n_detected))
                
                # 2. Optimización inteligente para visualización (Downsampling)
                # Si N es muy grande, mostramos 1 de cada 'skip' píxeles para no saturar memoria
                skip = 1
                if n_detected > 800: skip = 2
                if n_detected > 1500: skip = 4
                
                grid_vis = grid_final[::skip, ::skip]
                
                # 3. Corrección de Ejes: Generamos coordenadas reales para que la gráfica muestre 0-N
                # aunque estemos mostrando menos píxeles.
                real_x = np.arange(0, n_detected, skip)
                real_y = np.arange(0, n_detected, skip)
                
                fig_map = go.Figure(data=go.Heatmap(
                    z=grid_vis, 
                    x=real_x, # Ejes reales
                    y=real_y, # Ejes reales
                    colorscale='Hot'
                ))
                
                fig_map.update_layout(
                    title=f"Distribución de Temperatura Final (N={n_detected})",
                    xaxis_title="Eje X", 
                    yaxis_title="Eje Y",
                    yaxis=dict(scaleanchor="x", scaleratio=1), # Proporción cuadrada
                    plot_bgcolor='black'
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.error(f"⚠️ El archivo binario tiene un tamaño extraño ({num_doubles} datos) que no forma un cuadrado perfecto.")
                n_detected = 0 
                
        except Exception as e:
            st.error(f"Error leyendo el archivo de resultados: {e}")
            n_detected = 0
    else:
        st.info("No se encontró archivo de resultados. Ejecuta el análisis primero.")

    # ---------------------------------------------------------
    # 5. ANIMACIÓN DE EVOLUCIÓN
    # ---------------------------------------------------------
    st.divider()
    st.subheader(f"🎥 Evolución Temporal")

    # Solo habilitar si detectamos un archivo válido
    if n_detected > 0 and st.button("🎬 Generar Animación"):
        
        N_ANIM = n_detected 
        FRAMES = 25
        anim_data = []
        temp_dir = tempfile.mkdtemp()
        
        st.info(f"Renderizando {FRAMES} fotogramas para N={N_ANIM}...")
        
        try:
            shutil.copy("simulacion_app", os.path.join(temp_dir, "simulacion_app"))
            p_bar = st.progress(0)
            
            # Determinar factor de salto para optimización (mismo criterio que arriba)
            skip = 1
            if N_ANIM > 800: skip = 2     
            if N_ANIM > 1500: skip = 4    
            
            for i in range(1, FRAMES + 1):
                c_step = int(steps * (i/FRAMES))
                if c_step == 0: c_step = 1
                
                # Ejecutar simulación para el frame actual
                subprocess.run(["mpirun", "--allow-run-as-root", "-np", str(max_procs), "./simulacion_app", str(N_ANIM), str(N_ANIM), str(c_step), str(HEAT_SOURCE_ID)], cwd=temp_dir, capture_output=True)
                
                bin_p = os.path.join(temp_dir, FILENAME_DEFAULT)
                if os.path.exists(bin_p):
                    d = np.fromfile(bin_p, dtype=np.float64)
                    if d.size == N_ANIM**2: 
                        grid_full = d.reshape((N_ANIM, N_ANIM))
                        # Guardar versión optimizada
                        anim_data.append(grid_full[::skip, ::skip])

                p_bar.progress(i/FRAMES)
        finally:
            shutil.rmtree(temp_dir)
        
        if anim_data:
            # Coordenadas reales para los ejes de la animación
            real_x = np.arange(0, N_ANIM, skip)
            real_y = np.arange(0, N_ANIM, skip)

            # Frame base
            fig_anim = go.Figure(data=[go.Heatmap(z=anim_data[0], x=real_x, y=real_y, colorscale='Hot', zmin=0, zmax=1)])
            
            # Frames de animación
            fig_anim.frames = [
                go.Frame(data=[go.Heatmap(z=g, x=real_x, y=real_y)], name=f"fr{k}") 
                for k, g in enumerate(anim_data)
            ]
            
            fig_anim.update_layout(
                title=f"Animación de Propagación (N={N_ANIM})",
                xaxis_title="Eje X",
                yaxis_title="Eje Y",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                plot_bgcolor='black',
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(
                        label="▶️ Reproducir",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=50, redraw=True), 
                            fromcurrent=True,
                            transition=dict(duration=0) 
                        )]
                    )]
                )],
                sliders=[{
                    "steps": [
                        {
                            "args": [[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                            "label": f"{k}", 
                            "method": "animate"
                        } for k, f in enumerate(fig_anim.frames)
                    ],
                }]
            )
            st.plotly_chart(fig_anim, use_container_width=True)