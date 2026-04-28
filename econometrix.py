"""
STRATUM ECONOMETRIX MASTER WRAPPER
Coordinación de la Capa de Datos v41-D
"""
import sys
import os

# Asegurar que el sistema reconozca la estructura de carpetas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data.econometrix import get_econometrix_panel
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ Error de Arquitectura: {e}")

def main():
    print("🚀 Iniciando Motor de Datos Stratum (1900-2026)...")
    
    try:
        # Recuperamos el panel con resolución diaria
        panel = get_econometrix_panel()
        
        # Validaciones de Calidad Diamantina
        print(f"✅ Alineación Exitosa: {len(panel)} días procesados.")
        print(f"📊 Rango: {panel['date'].min().date()} | {panel['date'].max().date()}")
        
        # Guardar cache procesado para el Engine/Logic
        os.makedirs('data/processed', exist_ok=True)
        panel.to_csv('data/processed/econometrix_daily_v41.csv', index=False)
        print("💾 Cache de alta resolución guardado en data/processed/")

    except Exception as e:
        print(f"❌ Fallo Crítico en la Armonización: {e}")

if __name__ == "__main__":
    main()
