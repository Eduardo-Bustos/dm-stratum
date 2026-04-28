import os
import pandas as pd
from data.econometrix import EconometrixLoader
from engine.logic import StratumLogic
from engine.models import StratumStateModel
from scripts.citr import StratumVisualizer

def run_master_pipeline():
    print("💎 INICIANDO PIPELINE MAESTRO STRATUM v41-D (Edición Diamantina)...")
    
    # 1. CARGA (Econometría)
    loader = EconometrixLoader()
    df = loader.load_and_clean()
    print("🛠 Datos Econometrix 2000-2026 cargados.")

    # 2. MODELADO (Estado Logit & Theta/SG)
    model = StratumStateModel()
    df = model.get_system_metrics(df)
    print("⚖️ Ecuación de Estado Logit ejecutada.")

    # 3. INFERENCIA (Logic & CITR)
    engine = StratumLogic()
    # Integrar ML/DL/Stochastic según el logic.py desarrollado
    df_final = engine.process_state(df) 
    print("🧠 Inferencia Neural y Monte Carlo completada.")

    # 4. EXPORTACIÓN DE TABLAS (Justificación BIS)
    os.makedirs("outputs/tables", exist_ok=True)
    df_final.to_csv("outputs/tables/TABLA_JUSTIFICACION_v41D.csv", index=False)
    df_final.to_csv("data/processed/citr_panel.csv", index=False) # Para visualizer
    print("💾 Grabando resultados en Drive...")

    # 5. VISUALIZACIÓN (Reporte Erudito)
    viz = StratumVisualizer()
    reporte_path = viz.plot_magistral_dashboard(df_final)
    print(f"📈 Reporte Magistral generado: {reporte_path}")

    print("\n✅ PIPELINE COMPLETADO EXITOSAMENTE.")
    print(f"Veredicto Actual: {df_final['Regimen_v41D'].iloc[-1]}")

if __name__ == "__main__":
    run_master_pipeline()
