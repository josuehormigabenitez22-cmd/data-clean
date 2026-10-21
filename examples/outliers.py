import pandas as pd
import numpy as np

df = pd.DataFrame({
    'producto': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'precio': [25, 30, 28, 5000, 22, 31, 27]  # 5000 es sospechoso
})

# Método IQR (Interquartile Range) — solo Pandas, sin dependencias extra
Q1 = df['precio'].quantile(0.25)
Q3 = df['precio'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df['precio'] < limite_inferior) | (df['precio'] > limite_superior)]
print(f"Outliers detectados: {len(outliers)}")
print(outliers)

# Alternativa: Z-score manual (sin scipy, solo numpy)
media = df['precio'].mean()
std = df['precio'].std()
df['z_score'] = np.abs((df['precio'] - media) / std)
outliers_z = df[df['z_score'] > 3]

# Decidir: ¿eliminar o marcar?
df['es_outlier'] = (df['precio'] < limite_inferior) | (df['precio'] > limite_superior)