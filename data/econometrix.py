import pandas as pd
import numpy as np
import os
from scipy.interpolate import PchipInterpolator

class SecularDataEngine:
    """
    ENGINE: STRATUM ECONOMETRIX CORE (v41-D)
    Responsabilidad: Reconstrucción de la continuidad diaria 1900-2026.
    """
    def __init__(self, raw_path='data/raw/econometrix_master.csv'):
        self.raw_path = raw_path
        self.target_date = '2026-04-28'

    def harmonize_secular_data(self):
        """
        Transforma series históricas heterogéneas en un flujo diario homogéneo.
        """
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"💎 Error Stratum: No se detecta el Master en {self.raw_path}")

        # Carga con parsing de fechas erudito
        df = pd.read_csv(self.raw_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        # 1. Creación del Canvas Diario (1900 -> Abril 2026)
        full_index = pd.date_range(start=df.index.min(), end=self.target_date, freq='D')
        df_daily = df.reindex(full_index)

        # 2. Interpolación Diamantina (Preservación de Monotonicidad)
        # Usamos PCHIP para evitar el "overshooting" de los splines convencionales
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Identificamos puntos válidos para el interpolador
            valid = df[col].dropna()
            if len(valid) > 1:
                x_points = (valid.index - valid.index[0]).days.values
                y_points = valid.values
                interp_func = PchipInterpolator(x_points, y_points)
                
                # Generamos los puntos para el nuevo índice
                all_days = (full_index - valid.index[0]).days.values
                df_daily[col] = interp_func(all_days)

        # 3. Inyección de Rigidez Institucional (Concentración)
        # La concentración (C_top10) se trata como un step-function (ffill)
        if 'c_top10' in df_daily.columns:
            df_daily['c_top10'] = df_daily['c_top10'].ffill()

        return df_daily.reset_index().rename(columns={'index': 'date'})

def get_econometrix_panel():
    """Punto de acceso magistral para el Pipeline."""
    engine = SecularDataEngine()
    return engine.harmonize_secular_data()
