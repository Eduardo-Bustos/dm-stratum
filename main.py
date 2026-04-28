# main.py - Orquestador Estratégico Stratum v41-D
from data.econometrix import EconometrixLoader
from engine.logic import StratumLogic
import os

def ejecutar_pipeline_maestro():
    # 1. CARGA (Econometría)
    loader = EconometrixLoader()
    df_limpio = loader.load_and_clean()
    
    # 2. PROCESAMIENTO (Ingeniería de Índices)
    engine = StratumLogic()
    
    # Aplicamos la lógica fila por fila (o vectorizada) para determinar la fase
    df_limpio['coherence'] = engine.compute_system_coherence(df_limpio['clock_dispersion'])
    df_limpio['regimen_estratum'] = df_limpio.apply(engine.evaluate_phase, axis=1)
    
    # 3. VEREDICTO (CITR)
    # Identificamos el momento exacto de la "Pérdida de Equivalencia"
    df_limpio['CITR_Collapse'] = (df_limpio['regimen_estratum'] == "PHASE_III_TRANSMISSION").astype(int)
    
    # 4. SALIDA Y EXPORTACIÓN
    output_path = "outputs/tables/STRATUM_FINAL_UNIFICADO.csv"
    os.makedirs("outputs/tables", exist_ok=True)
    df_limpio.to_csv(output_path, index=False)
    
    # Reporte de War-Room
    ultimo = df_limpio.iloc[-1]
    print(f"--- REPORTE UNIFICADO STRATUM ---")
    print(f"Régimen Actual: {ultimo['regimen_estratum']}")
    print(f"Coherencia Crítica: {ultimo['coherence']:.4f}")
    print(f"CITR_Collapse: {'🔴 ACTIVADO' if ultimo['CITR_Collapse'] == 1 else '🟢 LATENTE'}")
    
    return df_limpio

# Ejecutar todo el sistema
# if __name__ == "__main__":
#    resultado = ejecutar_pipeline_maestro()
