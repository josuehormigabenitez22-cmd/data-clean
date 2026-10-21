import pandas as pd
import numpy as np

def limpiar_ventas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame de ventas aplicando los 4 pasos estándar.
    Retorna un DataFrame limpio y loguea las transformaciones.
    """
    df = df.copy()  # Nunca modifiques el DataFrame que te pasan
    filas_iniciales = len(df)
    
    # 1. Eliminar duplicados por ID
    df = df.drop_duplicates(subset=['pedido_id'], keep='last')
    print(f"  Duplicados eliminados: {filas_iniciales - len(df)}")
    
    # 2. Manejar nulos
    # Importe nulo = eliminar (no podemos inventar ventas)
    df = df.dropna(subset=['importe'])
    # Ciudad nula = marcar como desconocida
    df['ciudad'] = df['ciudad'].fillna('Desconocida')
    
    # 3. Corregir tipos
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['importe'] = pd.to_numeric(df['importe'], errors='coerce')
    df = df.dropna(subset=['fecha', 'importe'])  # Los que no se pudieron parsear
    
    # 4. Normalizar texto
    df['ciudad'] = df['ciudad'].str.strip().str.title()
    df['producto'] = df['producto'].str.strip().str.upper()
    
    filas_finales = len(df)
    print(f"  Resultado: {filas_iniciales} → {filas_finales} filas ({filas_iniciales - filas_finales} eliminadas)")
    
    return df.reset_index(drop=True)