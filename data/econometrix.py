import pandas as pd
import numpy as np
import os

def get_historical_data():
    """
    Capa Core Stratum v41. 
    Optimización de Tensores para Montecarlo y Prospectiva Sistémica.
    """
    # 1. LOCALIZACIÓN DEL DATASET (Prioridad v41 Final)
    filename = "STRATUM_v41_ECONOMETRICO_FINAL (1).csv"
    path = os.path.join("data", filename) if "data" not in os.getcwd() else filename
    
    if not os.path.exists(path):
        # Búsqueda recursiva de emergencia
        path = filename if os.path.exists(filename) else None
        
    if not path:
        return None

    try:
        # 2. CARGA Y NORMALIZACIÓN DINÁMICA
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        
        # Mapeo de Proposición 2 (Sincronización con logic.py)
        # Fragility se convierte en CP (Concentración/Presión estructural)
        mapping = {'date': 'date', 'SG': 'SG', 'ISI': 'ISI', 'Fragility': 'CP'}
        df.rename(columns=mapping, inplace=True)

        # 3. PREPARACIÓN PARA MONTECARLO Y VAR
        # Convertimos fechas y rellenamos huecos con interpolación lineal
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
        
        for col in ['SG', 'ISI', 'CP']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').interpolate().fillna(0)
            else:
                df[col] = 0.0

        # 4. AUDITORÍA DE CALIDAD (ADN del Dato)
        # Eliminamos infinitos que arruinarían las 100k iteraciones
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.reset_index(drop=True)
        
        return df

    except Exception as e:
        print(f"Error en Core Econometrix: {e}")
        return None
