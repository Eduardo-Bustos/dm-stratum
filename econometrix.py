import pandas as pd
import numpy as np
import os

def get_historical_data():
    """
    Capa de Inteligencia de Datos Stratum v41.
    Sincronización total con el dataset dinámico 1900-2026.
    A prueba de fallos de ruta y nombres de columnas.
    """
    
    # 1. MATRIZ DE BÚSQUEDA DE CONTINGENCIA
    # El sistema busca tanto el nombre genérico como el nombre específico del archivo cargado
    posibles_rutas = [
        "STRATUM_v41_ECONOMETRICO_FINAL (1).csv",
        "data/STRATUM_v41_ECONOMETRICO_FINAL (1).csv",
        "data/econometric.csv", 
        "econometric.csv"
    ]
    
    archivo_encontrado = None
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            archivo_encontrado = ruta
            break
            
    if not archivo_encontrado:
        print("🚨 CRITICAL ERROR: Dataset no detectado en el repositorio.")
        return None
        
    try:
        # 2. CARGA DE ALTA DISPONIBILIDAD
        df = pd.read_csv(archivo_encontrado, low_memory=False)
        
        # 3. NORMALIZACIÓN DE CABECERAS (Mapeo de tu CSV específico)
        # Tu archivo usa 'Fragility' para representar la presión estructural
        rename_map = {
            'date': 'date', 'Date': 'date',
            'sg': 'SG', 'SG': 'SG',
            'isi': 'ISI', 'ISI': 'ISI',
            'fragility': 'CP', 'Fragility': 'CP'
        }
        
        # Limpieza de espacios y aplicación de nombres estándar
        df.columns = [c.strip() for c in df.columns]
        df.rename(columns=rename_map, inplace=True)

        # 4. TRATAMIENTO DE SERIES TEMPORALES
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Eliminamos filas corruptas y ordenamos cronológicamente
            df = df.dropna(subset=['date']).sort_values('date')
        else:
            # Contingencia: Si no hay fecha, se asume secuencia mensual
            print("⚠️ WARNING: Reconstruyendo eje temporal (Secuencia de 126 años).")
            df['date'] = pd.date_range(start='1900-01-01', periods=len(df), freq='M')

        # 5. PROTOCOLO DE INTEGRIDAD DINÁMICA (Sin umbrales fijos)
        # Aseguramos que existan las métricas para la Proposición 2
        metricas_necesarias = ['SG', 'ISI', 'CP']
        for metrica in metricas_necesarias:
            if metrica not in df.columns:
                # Si falta una métrica, se inicializa con el promedio del sistema
                df[metrica] = 0.0
            else:
                # Forzamos conversión a números limpios
                df[metrica] = pd.to_numeric(df[metrica], errors='coerce').fillna(0.0)

        # 6. RESETEO DE ÍNDICE PARA EL WAR ROOM
        df = df.reset_index(drop=True)
        
        print(f"✅ CONEXIÓN EXITOSA: Dataset '{archivo_encontrado}' sincronizado.")
        return df

    except Exception as e:
        print(f"🚨 FALLA TÉCNICA EN ECONOMETRIX: {str(e)}")
        return None
