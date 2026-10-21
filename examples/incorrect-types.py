import pandas as pd

df = pd.DataFrame({
    'fecha': ['2024-01-15', '2024-02-20', '15/03/2024', '2024-04-10'],
    'importe': ['1500.50', '2300', 'N/A', '890.25'],
    'activo': ['si', 'no', 'si', 'SI'],
    'codigo_postal': [28001, 8001, 41001, 28020]
})

print("ANTES:")
print(df.dtypes)

# Convertir fecha (maneja formatos mixtos)
df['fecha'] = pd.to_datetime(df['fecha'], format='mixed', dayfirst=True, errors='coerce')

# Convertir importe (primero reemplazar texto, luego convertir)
df['importe'] = pd.to_numeric(df['importe'], errors='coerce')

# Convertir booleano normalizado
df['activo'] = df['activo'].str.lower().map({'si': True, 'no': False})

# Código postal como string (para preservar ceros)
df['codigo_postal'] = df['codigo_postal'].astype(str).str.zfill(5)

print("\nDESPUÉS:")
print(df.dtypes)
print(df)