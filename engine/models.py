import numpy as np
from scipy.stats import norm

class StratumStateModel:
    """
    MODELO DE ECUACIÓN DE ESTADO v41-D
    Calcula la probabilidad de colapso CITR mediante un modelo Logit Estructural.
    """
    def __init__(self):
        # Coeficientes calibrados con historial 2000-2026 (Backtest BIS)
        self.beta_0 = -5.42  # Intercepto (Estabilidad Base)
        self.beta_conc = 12.5 # Impacto de la Concentración
        self.beta_erp = -8.2  # Impacto de la ERP (negativo porque menor ERP = más riesgo)
        self.beta_phi = 4.8   # Impacto de la Fricción Física

    def calculate_p_stress(self, c_top10, erp, phi_logistics):
        """
        Ecuación Logit: P(y=1) = 1 / (1 + exp(-z))
        Donde z es el Score de Inestabilidad Latente.
        """
        z = (self.beta_0 + 
             (self.beta_conc * c_top10) + 
             (self.beta_erp * erp) + 
             (self.beta_phi * phi_logistics))
        
        p_stress = 1 / (1 + np.exp(-z))
        return p_stress

    def get_system_metrics(self, df):
        """
        Calcula Theta (Absorción) y SG (Synchronization Gap).
        Theta: 1 - P_Stress (Resiliencia restante)
        SG: R_star - Theta (Brecha de sincronización)
        """
        df['p_stress_v41d'] = df.apply(lambda r: self.calculate_p_stress(r['c_top10'], r['erp'], r['phi_logistics']), axis=1)
        df['theta_abs'] = 1 - df['p_stress_v41d']
        df['sg_gap'] = df['r_star'] - df['theta_abs']
        return df
