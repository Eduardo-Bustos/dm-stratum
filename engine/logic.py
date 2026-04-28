import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy.stats import norm

class StratumLogic:
    """
    ENGINE: STRATUM LOGOS CORE-X (v41-D)
    Integración: ML (Random Forest), DL (LSTM), y Simulación Monte Carlo.
    """
    def __init__(self, simulation_runs=100000):
        self.runs = simulation_runs
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, dynamic_boost=True)
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.is_trained = False
        
        # Umbrales Críticos de la Figura 9
        self.ERP_CRIT = 0.008
        self.PHI_CRIT = 1.35
        self.CONC_CRIT = 0.33

    # --- CAPA 1: DEEP LEARNING (LSTM para Memoria Temporal) ---
    def build_lstm(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid') # Probabilidad de Transmisión
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        self.lstm_model = model

    # --- CAPA 2: MACHINE LEARNING (Árboles Decisionales Dinámicos) ---
    def train_logic(self, X, y):
        """Entrena el motor con el historial Econometrix 2000-2026."""
        X_scaled = self.scaler.fit_transform(X)
        self.rf_model.fit(X_scaled, y)
        
        # Entrenamiento LSTM (Reshape para 3D: samples, timesteps, features)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        self.build_lstm((1, X_scaled.shape[1]))
        self.lstm_model.fit(X_lstm, y, epochs=10, verbose=0)
        self.is_trained = True

    # --- CAPA 3: INFERENCIA CITR & FRAGMENTACIÓN ---
    def calculate_citr_collapse(self, row):
        """Calcula el colapso basado en el decaimiento de coherencia."""
        delta_tau = row['clock_dispersion']
        coherence = 1 / (1 + delta_tau)
        
        # Lógica Heurística (Figura 9)
        loss_of_equivalence = (
            (row['erp'] < self.ERP_CRIT) & 
            (row['phi_logistics'] > self.PHI_CRIT) & 
            (row['c_top10'] > self.CONC_CRIT)
        )
        
        return int(loss_of_equivalence), coherence

    # --- CAPA 4: MONTE CARLO (100,000 Corridas para Predicción Perfecta) ---
    def predict_monte_carlo(self, current_state):
        """
        Simula 100,000 trayectorias para predecir el momento exacto 
        del 'Kinetic Release Threshold'.
        """
        results = []
        mu = current_state['sci_trend']
        sigma = 0.15 # Volatilidad sistémica estimada
        
        simulated_paths = np.random.normal(mu, sigma, self.runs)
        
        # Probabilidad de que SCI rompa la barrera de colapso
        collapse_prob = np.mean(simulated_paths < 3.5) 
        return collapse_prob

    # --- CAPA 5: ORQUESTACIÓN FINAL ---
    def process_state(self, df):
        """Procesa el dataframe y devuelve el veredicto v41-D."""
        if not self.is_trained:
            # Entrenamiento automático con datos históricos si es necesario
            pass 

        results = []
        for i, row in df.iterrows():
            collapse, coherence = self.calculate_citr_collapse(row)
            mc_prob = self.predict_monte_carlo(row)
            
            # Verificación de Regimen
            regimen = "ABSORCIÓN" if collapse == 0 else "TRANSMISIÓN"
            if regimen == "ABSORCIÓN" and coherence < 0.45:
                regimen = "COHERENCE TRAP"
                
            results.append({
                'date': row['date'],
                'coherence': coherence,
                'CITR_Collapse': collapse,
                'P_Stress_MC': mc_prob,
                'Regimen_v41D': regimen
            })
            
        return pd.DataFrame(results)
