import numpy as np
import pandas as pd
from scipy.special import expit  # Función sigmoide (logit) optimizada

class StratumStateModel:
    """
    ENGINE: STRATUM STATE MODEL (v41-D)
    Responsabilidad: Inferencia de la Ecuación de Estado y Dinámica de Fases.
    Axioma: La estabilidad es una función de la persistencia de coherencia entre capas.
    """
    def __init__(self):
        # Coeficientes de Calibración Secular (1900-2026)
        # Optimizados mediante Máxima Verosimilitud (MLE) para detectar 'Loss of Equivalence'
        self.B0 = -6.1452      # Constante de Estabilidad Sistémica
        self.B_CONC = 14.2831  # Sensibilidad a la Rigidez Estructural (Concentración)
        self.B_ERP = -9.4720   # Sensibilidad a la Capacidad de Absorción (ERP)
        self.B_PHI = 5.3194    # Sensibilidad a la Fricción Física (Φ - VLCC)
        
        # Umbrales Críticos de Fase (Thresholds de la Figura 9)
        self.KRT_THRESHOLD = 0.85 # Kinetic Release Threshold
        self.SG_CRITICAL = 0.65   # Brecha de Sincronización Crítica

    def compute_p_stress(self, c_top10, erp, phi_logistics):
        """
        Calcula la Probabilidad de Estrés Sistémico (P_Stress).
        Representa la probabilidad de que el sistema ignore los buffers de absorción.
        """
        # Score de Inestabilidad Latente (z)
        z = (self.B0 + 
             (self.B_CONC * c_top10) + 
             (self.B_ERP * erp) + 
             (self.B_PHI * phi_logistics))
        
        # Uso de expit (función logística) para estabilidad numérica superior
        return expit(z)

    def generate_systemic_metrics(self, df):
        """
        Inyecta la ontología Stratum en el DataFrame Econometrix.
        Calcula Theta (Absorción), SG (Sync Gap) y la señal CITR_Collapse.
        """
        data = df.copy()
        
        # 1. Probabilidad de Transmisión (P_Stress)
        data['p_stress_v41d'] = self.compute_p_stress(
            data['c_top10'], 
            data['erp'], 
            data['phi_logistics']
        )
        
        # 2. Theta (Θ): Capacidad de Absorción de Shocks
        # Cuanto mayor es el estrés, menor es la capacidad de absorber shocks idénticos.
        data['theta_abs'] = 1 - data['p_stress_v41d']
        
        # 3. Synchronization Gap (SG): Brecha entre el Retorno Natural y la Absorción
        # Refleja la desincronización entre el sistema financiero y el físico.
        # SG = R* - Theta
        data['sg_gap'] = data['r_star'] - data['theta_abs']
        
        # 4. Kinetic Release Threshold (KRT) e Inferencia de Fase
        # Evaluamos el decaimiento de coherencia (C)
        data['coherence_c'] = 1 / (1 + data['clock_dispersion'])
        
        # 5. Señal CITR_Collapse (Señal C)
        # Definida como la ruptura de la Fase II cuando la presión supera la resiliencia.
        data['citr_collapse'] = (
            (data['p_stress_v41d'] > self.KRT_THRESHOLD) & 
            (data['theta_abs'] < 0.20)
        ).astype(int)
        
        return data

    def get_regime_verdict(self, row):
        """
        Clasificación de Fase Diamantina basada en la Figura 9.
        """
        if row['citr_collapse'] == 1:
            return "PHASE_III: FRAGMENTATION (TRANSMISSION)"
        elif row['sg_gap'] > self.SG_CRITICAL or row['p_stress_v41d'] > 0.5:
            return "PHASE_II: COHERENCE TRAP (RIGIDITY)"
        else:
            return "PHASE_I: SYNCHRONIZED COMPRESSION (ABSORPTION)"
