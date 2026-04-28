import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import seaborn as sns

# Configuraciones de Estilo "Think-Tank"
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

class StratumVisualizer:
    """
    GENERADOR DE EVIDENCIA SISTÉMICA - STRATUM V41-D
    Produce el 'War-Room Dashboard' para el análisis de colapso de coherencia.
    """
    def __init__(self, output_path="outputs/figures/citr/"):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def plot_magistral_dashboard(self, df):
        """
        Crea una infografía de alta resolución comparando las capas 
        Financiera, Física e Institucional.
        """
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = gridspec.GridSpec(3, 2, figure=fig)
        
        # --- 1. CAPA INSTITUCIONAL: Concentración vs Estabilidad ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['date'], df['c_top10'], color='#1B4F72', lw=2, label='Top-10 Weight')
        ax1.axhline(0.33, color='red', ls='--', alpha=0.6, label='C-Crit Threshold')
        ax1.set_title("I. RIGIDEZ ESTRUCTURAL (CONCENTRACIÓN S&P 500)")
        ax1.legend(loc='upper left')
        
        # --- 2. CAPA FÍSICA: Fricción Logística (VLCC) ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.fill_between(df['date'], df['phi_logistics'], 1.0, color='#D35400', alpha=0.3)
        ax2.plot(df['date'], df['phi_logistics'], color='#A04000', lw=1.5)
        ax2.set_title("II. FRICCIÓN FÍSICA Φ (SURGE DE VLCC VACÍOS)")
        
        # --- 3. CAPA FINANCIERA: Compresión de la ERP ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df['date'], df['erp'], color='#145A32', lw=2)
        ax3.axhline(0.008, color='darkred', ls=':', label='Absorción Zero')
        ax3.set_title("III. EQUITY RISK PREMIUM (BUFFER DE ABSORCIÓN)")
        ax3.invert_yaxis() # ERP baja = Riesgo alto
        
        # --- 4. ENGINE: Coherencia vs KRT (Liberación Cinética) ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df['date'], df['coherence'], color='#7D3C98', label='Coherence (C)')
        ax4.plot(df['date'], df['krt_intensity'], color='#CB4335', ls='-', label='KRT Intensity')
        ax4.set_title("IV. DINÁMICA DE COHERENCIA Y ENERGÍA CINÉTICA")
        ax4.legend()

        # --- 5. VEREDICTO: Probabilidad Monte Carlo (GPU Output) ---
        ax5 = fig.add_subplot(gs[2, :])
        sns.lineplot(x='date', y='P_Stress_MC', data=df, ax=ax5, color='black', lw=2)
        ax5.fill_between(df['date'], df['P_Stress_MC'], 0.85, 
                         where=(df['P_Stress_MC'] >= 0.85), color='red', alpha=0.4, label='Phase Transition Zone')
        ax5.set_title("V. PROBABILIDAD DE COLAPSO SISTÉMICO (MONTE CARLO 100K PATHS)")
        ax5.set_ylim(0, 1.1)
        ax5.legend()

        # Metadatos de la Firma
        fig.suptitle(f"STRATUM SYSTEMIC FRAMEWORK - DIAGNÓSTICO ESTRATÉGICO v41-D\nData Final: {df['date'].iloc[-1].strftime('%Y-%m-%d')}", 
                     fontsize=18, fontweight='bold')
        
        # Guardar en calidad diamantina
        filename = f"STRATUM_MASTER_DIAGNOSTIC_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        plt.savefig(self.output_path + filename, dpi=600, format='pdf', bbox_inches='tight')
        plt.show()
        return filename

# Lógica de Ejecución Automática
if __name__ == "__main__":
    # Importación de los motores que desarrollamos anteriormente
    from engine.logic import StratumLogic
    from data.econometrix import EconometrixLoader
    
    print("💎 Generando Reporte Diamante...")
    
    # 1. Carga
    loader = EconometrixLoader()
    df = loader.load_and_clean()
    
    # 2. Inferencia
    engine = StratumLogic()
    df_results = engine.process_state(df)
    
    # 3. Visualización
    viz = StratumVisualizer()
    reporte = viz.plot_magistral_dashboard(df_results)
    
    print(f"✅ Reporte Magistral generado: {reporte}")
