import sys
import os

# Aseguramos que el sistema vea la carpeta /data para importar el Core
sys.path.append(os.path.join(os.getcwd(), 'data'))

try:
    # Intenta importar la lógica extendida del archivo en /data
    from data.econometrix import get_historical_data as get_core_data
    def get_historical_data():
        return get_core_data()
except ImportError:
    # Si falla la importación (ej. en Docker), ejecuta la carga directa de emergencia
    import pandas as pd
    def get_historical_data():
        path = "data/STRATUM_v41_ECONOMETRICO_FINAL (1).csv"
        if os.path.exists(path):
            return pd.read_csv(path, low_memory=False)
        return None
