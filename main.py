import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from econometrix import get_historical_data
from engine.logic import StratumEngine
import datetime

# --- CONFIGURACIÓN DE INTERFAZ DE ALTA FIDELIDAD ---
st.set_page_config(
    page_title="Stratum Última | War Room v41",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS Erudito: Minimalismo institucional
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1a1c23; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    div[data-testid="stMetricValue"] > div { font-size: 28px; color: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🛡️ Stratum Última: War Room v41")
    st.caption("Ecosistema de Análisis de Riesgo Sistémico y Transición de Fase | Proposición 2")

    # --- INGESTA Y SINCRONIZACIÓN ---
    with st.spinner("Sincronizando Capa Econometrix..."):
        df_raw = get_historical_data()

    if df_raw is None or df_raw.empty:
        st.error("🚨 Error Crítico: No se pudo establecer el flujo de datos. Verifique 'data/econometric.csv'.")
        return

    # --- SIDEBAR: CONTROL DE SENSIBILIDAD DINÁMICA ---
    st.sidebar.header("Parámetros del Ecosistema")
    
    # En lugar de un Tau fijo, controlamos la sensibilidad por percentiles
    p_threshold = st.sidebar.slider(
        "Sensibilidad Crítica (Percentil)", 
        min_value=0.70, max_value=0.99, value=0.90, step=0.01,
        help="Ajusta el umbral dinámico basado en la distribución histórica (P90 por defecto)."
    )

    # Selector de Horizonte Temporal
    min_date = df_raw['date'].min().to_pydatetime()
    max_date = df_raw['date'].max().to_pydatetime()
    
    start_time, end_time = st.sidebar.slider(
        "Horizonte Temporal",
        min_value=min_date, max_value=max_date,
        value=(min_date, max_date),
        format="YYYY"
    )

    # --- PROCESAMIENTO DE LÓGICA LOGOS ---
    engine = StratumEngine(sensitivity_p=p_threshold)
    
    # Filtrado por ventana temporal
    mask = (df_raw['date'] >= start_time) & (df_raw['date'] <= end_time)
    df_filtered = df_raw.loc[mask].copy()

    # Ejecución de la Proposición 2
    df_processed, kpis = engine.process_system_state(df_filtered)

    # --- DASHBOARD DE ESTADOS CRÍTICOS ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "normal" if not kpis['trigger_active'] else "inverse"
        st.metric("Estado del Sistema", kpis['regime'], delta=None, delta_color=color)
    
    with col2:
        st.metric("Brecha SG (Actual)", f"{kpis['lambda_stat']:.4f}", 
                  f"Umbral: {kpis['dynamic_tau']:.4f}")
    
    with col3:
        st.metric("Saturación ISI", f"{kpis['lambda_threshold']:.2%}", 
                  f"Momentum: {kpis['momentum']:.4f}")
        
    with col4:
        st.metric("Fragilidad CP", f"{kpis['structural_concentration']:.4f}")

    # --- VISUALIZACIÓN GEOMÉTRICA DEL RIESGO ---
    st.subheader("Análisis de Estabilidad Sistémica")
    
    fig = go.Figure()

    # Serie de Brecha (SG)
    fig.add_trace(go.Scatter(
        x=df_processed['date'], y=df_processed['SG'].abs(),
        name="Magnitud SG", line=dict(color='#00d4ff', width=1.5),
        fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
    ))

    # Umbral Dinámico (Sustituye al 0.3492 fijo)
    fig.add_trace(go.Scatter(
        x=df_processed['date'], y=[kpis['dynamic_tau']] * len(df_processed),
        name="Umbral Dinámico (τ)",
        line=dict(color='rgba(255, 75, 75, 0.8)', width=2, dash='dot')
    ))

    # Marcadores de "Acute Stress"
    stress_events = df_processed[df_processed['threshold_breach']]
    if not stress_events.empty:
        fig.add_trace(go.Scatter(
            x=stress_events['date'], y=stress_events['SG'].abs(),
            mode='markers', name='Puntos de Ruptura',
            marker=dict(color='#ff4b4b', size=6)
        ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- AUDITORÍA DE DATOS ---
    with st.expander("Ver Matriz de Datos Sincronizada"):
        st.dataframe(df_processed.sort_values('date', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
