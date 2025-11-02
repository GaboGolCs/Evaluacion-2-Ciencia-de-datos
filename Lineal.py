import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("nuevoCSV.csv", sep=";", encoding="utf-8")

# contar cuántos casos se atendieron por día
casos_por_dia = (data.groupby(["FECHA_INGRESO", "SUCURSAL"]).size().reset_index(name="NUM_CASOS_DIA"))

# se crean columnas binarias
sucursales_codificadas = pd.get_dummies(casos_por_dia["SUCURSAL"], prefix="SUCURSAL")

# Unir las columnas
dataset_final = pd.concat([casos_por_dia, sucursales_codificadas], axis=1)


#solo usamos las columnas binarias para predecir
X = dataset_final.drop(columns=["FECHA_INGRESO", "SUCURSAL", "NUM_CASOS_DIA"])
# variable objetivo numero de casos por dia
y = dataset_final["NUM_CASOS_DIA"]


X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_regresion = LinearRegression()
modelo_regresion.fit(X_entrenamiento, y_entrenamiento)

predicciones = modelo_regresion.predict(X_prueba)

error_medio_cuadratico = mean_squared_error(y_prueba, predicciones)
raiz_error_medio = np.sqrt(error_medio_cuadratico)
r2 = r2_score(y_prueba, predicciones)

print("Evaluacion del modelo de Regresion Lineal:")
print(f"RMSE: {raiz_error_medio:.2f} -> En promedio, las predicciones se desvian en {raiz_error_medio:.2f} casos.")
print(f"R2: {r2:.2%} -> El modelo explica el {r2:.2%} de la variacion total en los casos diarios.\n")
# ... y el título del gráfico
plt.title("Regresion Lineal - Prediccion de numero de casos diarios por sucursal")


# coeficientes del modelo
importancia_sucursales = pd.DataFrame({
    "Sucursal": X.columns,
    "Coeficiente": np.round(modelo_regresion.coef_, 2)
}).sort_values(by="Coeficiente", ascending=False)

print("Sucursales con mayor número de casos promedio:")
print(importancia_sucursales.head(10))
print("\nSucursales con menor número de casos promedio:")
print(importancia_sucursales.tail(10))

# Gráfica de dispersión para visualizar las predicciones vs los valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_prueba, predicciones, alpha=0.6, color="royalblue", label="Datos de prueba")
plt.plot(
    [y_prueba.min(), y_prueba.max()],
    [y_prueba.min(), y_prueba.max()],
    'r--',
    lw=2,
    label="Línea ideal (y = x)"
)
plt.xlabel("Casos reales por día")
plt.ylabel("Casos predichos por el modelo")
plt.title("Regresión Lineal — Predicción de número de casos diarios por sucursal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

