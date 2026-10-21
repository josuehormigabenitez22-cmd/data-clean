import pandas as pd

df = pd.DataFrame({
    'pedido_id': [1, 2, 3, 2, 4, 3],
    'cliente': ['Ana', 'Carlos', 'Lucía', 'Carlos', 'Pedro', 'Lucía'],
    'importe': [100, 250, 75, 250, 180, 75],
    'fecha': ['2024-01-15', '2024-01-16', '2024-01-16', '2024-01-16', '2024-01-17', '2024-01-16']
})

# Detectar duplicados
print(f"Duplicados exactos: {df.duplicated().sum()}")
print(f"Duplicados por pedido_id: {df.duplicated(subset=['pedido_id']).sum()}")

# Eliminar duplicados exactos
df_limpio = df.drop_duplicates()

# Eliminar duplicados por clave, quedándose con el primero
df_limpio = df.drop_duplicates(subset=['pedido_id'], keep='first')

# Eliminar duplicados quedándose con el último (ej: la versión más reciente)
df_limpio = df.drop_duplicates(subset=['pedido_id'], keep='last')

print(f"Antes: {len(df)} filas → Después: {len(df_limpio)} filas")