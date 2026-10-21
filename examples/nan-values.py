import pandas as pd
import numpy as np

df = pd.DataFrame({
    'cliente': ['Ana', 'Carlos', None, 'Pedro', 'Ana'],
    'email': ['ana@mail.com', None, 'lucia@mail.com', 'pedro@mail.com', 'ana@mail.com'],
    'compras': [5, 12, None, 3, 5],
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', None, 'Madrid']
})

# Detectar nulos
print(df.isnull().sum())  # Nulos por columna
print(f"Total nulos: {df.isnull().sum().sum()}")
print(f"% filas con algún nulo: {df.isnull().any(axis=1).mean()*100:.1f}%")

# ESTRATEGIA 1: Eliminar filas con nulos (si son pocas)
df_sin_nulos = df.dropna()

# ESTRATEGIA 2: Rellenar con un valor por defecto
df['ciudad'] = df['ciudad'].fillna('Desconocida')
df['compras'] = df['compras'].fillna(df['compras'].median())

# ESTRATEGIA 3: Rellenar con la media/mediana (solo numéricos)
df['compras'] = df['compras'].fillna(df['compras'].median())