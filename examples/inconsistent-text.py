import pandas as pd

df = pd.DataFrame({
    'ciudad': [' Madrid ', 'madrid', 'BARCELONA', 'Barcleona', 'Madrid', 'barcelona'],
    'ventas': [100, 200, 150, 80, 300, 250]
})

# Paso 1: strip (quitar espacios) + lower (minúsculas)
df['ciudad_clean'] = df['ciudad'].str.strip().str.lower()

# Paso 2: corregir typos conocidos con replace/map
correcciones = {
    'barcleona': 'barcelona',
    'madird': 'madrid',
}
df['ciudad_clean'] = df['ciudad_clean'].replace(correcciones)

# Ahora el groupby funciona bien
print(df.groupby('ciudad_clean')['ventas'].sum())