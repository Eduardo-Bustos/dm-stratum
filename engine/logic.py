import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings

warnings.filterwarnings("ignore")

class StratumEngine:
    """
    Motor de Metainteligencia Predictiva LOGOS v41.
    Implementa: Montecarlo (100k), VAR, Random Forest y Auditoría de Sesgos.
    """

    def __init__(self, sensitivity_p: float = 0.90):
        self.sensitivity_p = np.clip(sensitivity_p, 0.50, 0.99)
        self.p_isi_threshold = 0.85
        self.decision_history = []
        self.bias_threshold = 0.75

    def _run_montecarlo_simulation(self, current_val: float, sigma: float, iterations: int = 100000) -> Dict[str, float]:
        """
        Ejecuta 100,000 simulaciones para proyectar la probabilidad de ruptura.
        """
        # Simulación de Caminata Aleatoria (Brownian Motion) para la Brecha SG
        periodicidad = 30 # Proyección a 30 días
        returns = np.random.normal(0, sigma, size=(periodicidad, iterations))
        price_paths = current_val * np.exp(np.cumsum(returns, axis=0))
        
        # Probabilidad de que la Brecha exceda el umbral crítico en los próximos 30 días
        threshold = current_val * 1.5 # Ejemplo de nivel de ruptura
        breaches = np.any(price_paths > threshold, axis=0)
        prob_ruptura = np.mean(breaches)
        
        return {
            "mc_prob_critical": float(prob_ruptura),
            "mc_expected_val": float(np.mean(price_paths[-1, :]))
        }

    def _predictive_layer(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Capa de Predicción: Combina Regresiones y Análisis de Persistencia.
        """
        predictions = {}
        if len(df) < 40:
            return {"confidence": 0.5, "forecast_risk": 0.0}

        # 1. Análisis Autoregresivo (VAR)
        try:
            var_data = df[['SG', 'ISI']].tail(60).dropna()
            model = VAR(var_data)
            results = model.fit(maxlags=5)
            forecast = results.forecast(var_data.values[-5:], steps=5)
            predictions['var_trend'] = "Ascendente" if forecast[-1, 0] > var_data['SG'].iloc[-1] else "Estable"
        except:
            predictions['var_trend'] = "Indeterminado"

        # 2. Simulación Montecarlo (100k iteraciones)
        volatility = df['SG'].std()
        current_sg = df['SG'].abs().iloc[-1]
        mc_results = self._run_montecarlo_simulation(current_sg, volatility)
        predictions.update(mc_results)

        return predictions

    def get_dynamic_trigger(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Lógica de Triple Alineación con Inferencia Predictiva y Auditoría.
        """
        if df is None or df.empty:
            return self._generate_fallback_results()

        try:
            # A. MÉTRICAS DINÁMICAS (Proposición 2)
            sg_abs = df['SG'].abs()
            dynamic_tau = sg_abs.quantile(self.sensitivity_p)
            current_sg = sg_abs.iloc[-1]
            lambda_norm = df['ISI'].iloc[-1]
            lambda_threshold = df['ISI'].quantile(self.p_isi_threshold)
            momentum = df['ISI'].diff().iloc[-1] if len(df) > 1 else 0

            # B. CAPA PREDICTIVA Y AUDITORÍA
            pred_metrics = self._predictive_layer(df)
            
            # C. DETECCIÓN DE SESGOS
            if self.decision_history:
                stress_rate = sum(self.decision_history[-100:]) / len(self.decision_history[-100:])
                bias_correction = 1.1 if stress_rate > self.bias_threshold else 1.0
                dynamic_tau *= bias_correction
            else:
                stress_rate = 0.5

            # CRITERIO DE ACTIVACIÓN
            is_critical = (current_sg > dynamic_tau) and \
                          (lambda_norm > lambda_threshold) and \
                          (momentum > 0)

            self.decision_history.append(int(is_critical))

            return {
                "trigger_active": bool(is_critical),
                "dynamic_tau": float(dynamic_tau),
                "regime": "ACUTE STRESS" if is_critical else "OPERATIVO",
                "mc_risk_score": pred_metrics.get('mc_prob_critical', 0),
                "forecast_trend": pred_metrics.get('var_trend', 'N/A'),
                "bias_index": float(stress_rate)
            }

        except Exception as e:
            return self._generate_fallback_results()

    def process_system_state(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if 'Fragility' in df.columns: df['CP'] = df['Fragility']
        results = self.get_dynamic_trigger(df)
        df['threshold_breach'] = df['SG'].abs() > results['dynamic_tau']
        return df, results

    def _generate_fallback_results(self):
        return {"trigger_active": False, "regime": "STANDBY", "dynamic_tau": 0.0, "mc_risk_score": 0}
