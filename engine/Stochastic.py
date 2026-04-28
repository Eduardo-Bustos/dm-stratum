import cupy as cp # NumPy para GPU
import numpy as np

class StratumStochastic:
    """
    Simulador Estocástico de Alto Rendimiento (HPS).
    Ejecuta simulaciones de Monte Carlo paralelas para detectar el Kinetic Release Threshold.
    """
    def __init__(self, runs=100000):
        self.runs = runs

    def simulate_krt_paths(self, current_sci, volatility, time_horizon=20):
        """
        Simula trayectorias futuras del SCI para predecir colapsos CITR.
        """
        # Transferencia de parámetros a la GPU
        sci_gpu = cp.array(current_sci)
        vol_gpu = cp.array(volatility)
        
        # Generación de ruido gaussiano masivo en GPU (100k corridas x Horizonte)
        noise = cp.random.normal(0, 1, (self.runs, time_horizon))
        
        # Geometrical Brownian Motion para el SCI
        # dSCI = mu*SCI*dt + sigma*SCI*dW
        dt = 1/252 # Escala diaria
        drift = -0.05 # Sesgo hacia la pérdida de coherencia en Fase II
        
        # Cálculo vectorial paralelo
        paths = sci_gpu * cp.exp(cp.cumsum((drift - 0.5 * vol_gpu**2) * dt + 
                                         vol_gpu * cp.sqrt(dt) * noise, axis=1))
        
        # Umbral de Colapso (C_crit) definido en Figura 9
        collapse_threshold = 3.5
        probabilities = cp.mean(cp.any(paths < collapse_threshold, axis=1))
        
        return float(probabilities), cp.asnumpy(paths) # Retornar prob y muestra
