import os
import sys
import pandas as pd
from datetime import datetime
import logging

# =============================================================================
# CONFIGURACIÓN DE RUTAS Y ENTORNO (Zero-Footprint Architecture)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Configuración de Logging Erudito
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | STRATUM-v41D | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Importaciones de Módulos Core
try:
    from data.econometrix import SecularDataEngine
    from engine.models import StratumStateModel
    from engine.logic import StratumLogic
    from scripts.citr import StratumVisualizer
except ImportError as e:
    logger.error(f"Falla de Integridad Estructural: {e}")
    sys.exit(1)

def run_master_pipeline():
    """
    ORQUESTADOR SUPREMO STRATUM v41-D
    Ejecuta la secuencia de inferencia probabilística y generación de evidencia.
    """
    start_time = datetime.now()
    logger.info("INICIANDO PIPELINE MAESTRO: Fase de Diagnóstico Secular")

    try:
        # 1. CAPA DE DATOS: Armonización 1900-2026
        # ---------------------------------------------------------------------
        raw_data_path = os.path.join(BASE_DIR, 'data', 'raw', 'econometrix_master.csv')
        data_engine = SecularDataEngine(raw_path=raw_data_path)
        df_daily = data_engine.harmonize_secular_data()
        logger.info(f"Capa de Datos: {len(df_daily)} registros diarios armonizados exitosamente.")

        # 2. CAPA DE MODELADO: Ecuación de Estado Logit
        # ---------------------------------------------------------------------
        state_model = StratumStateModel()
        df_enriched = state_model.generate_systemic_metrics(df_daily)
        logger.info("Capa de Modelado: Ecuación de Estado Logit inyectada (Theta & SG).")

        # 3. CAPA DE INFERENCIA: Neural & Stochastic (100k paths)
        # ---------------------------------------------------------------------
        logic_engine = StratumLogic()
        df_final = logic_engine.process_state(df_enriched)
        
        # Aplicación del veredicto final basado en la ontología Stratum
        df_final['Regimen_v41D'] = df_final.apply(state_model.get_regime_verdict, axis=1)
        logger.info("Capa de Inferencia: Proceso Neural y Monte Carlo finalizado.")

        # 4. CAPA DE SALIDA: Justificación Técnica y Visualización
        # ---------------------------------------------------------------------
        # Exportación de Tabla de Justificación
        tables_path = os.path.join(BASE_DIR, 'outputs', 'tables')
        os.makedirs(tables_path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        table_name = f"TABLA_JUSTIFICACION_v41D_{timestamp}.csv"
        df_final.to_csv(os.path.join(tables_path, table_name), index=False)
        
        # Generación de Reporte Diamante (PDF)
        figures_path = os.path.join(BASE_DIR, 'outputs', 'figures', 'citr')
        viz = StratumVisualizer(output_path=figures_path)
        reporte_pdf = viz.plot_magistral_dashboard(df_final)

        # FINALIZACIÓN
        # ---------------------------------------------------------------------
        duration = datetime.now() - start_time
        logger.info(f"PIPELINE COMPLETADO en {duration.total_seconds():.2f}s")
        logger.info(f"VEREDICTO ACTUAL: {df_final['Regimen_v41D'].iloc[-1]}")
        logger.info(f"Documento Maestro: {reporte_pdf}")

    except Exception as e:
        logger.critical(f"COLAPSO DEL PIPELINE: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_master_pipeline()
