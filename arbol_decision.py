import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Cargar datos
df = pd.read_csv("nuevoCSV.csv", sep=";", encoding="utf-8")

# --- Filtramos las instituciones que tengan al menos 100 casos ---
instituciones_frecuentes = df["INSTITUCION"].value_counts()
instituciones_validas = instituciones_frecuentes[instituciones_frecuentes >= 100].index
df_filtrado = df[df["INSTITUCION"].isin(instituciones_validas)].copy()

print(f"Instituciones totales: {df['INSTITUCION'].nunique()}")
print(f"Instituciones con suficientes datos: {len(instituciones_validas)}")

# --- Variables de prediccion (X) y objetivo (y) ---
x = df_filtrado[["MATERIA", "SUBMATERIA", "TIPO_CONSULTA", "SUCURSAL"]].copy()
y = df_filtrado["INSTITUCION"].astype(str).copy()

# --- Codificación de texto a números ---
encoders = {}
x_encoded = x.copy()
for columna in x.columns:
    le = LabelEncoder()
    x_encoded[columna] = le.fit_transform(x[columna].astype(str))
    encoders[columna] = le

le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)
class_names = le_y.classes_

# --- División de datos (70% entrenamiento, 30% prueba) ---
X_entrenamiento, X_testeo, Y_entrenamiento, Y_testeo = train_test_split(x_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# --- Entrenamiento del modelo ---
model = DecisionTreeClassifier(
    max_depth=3,         # Limita la profundidad del árbol
    random_state=42,     # Para reproducibilidad
    criterion='entropy',   # mide la pureza de los nodos
    min_samples_split=50, # Mínimo de muestras para dividir un nodo
    min_samples_leaf=20    # Obliga a que cada hoja tenga al menos 20 muestras
)
model.fit(X_entrenamiento, Y_entrenamiento)

# --- Visualización del árbol ---
plt.figure(figsize=(35, 20))   #tamaño de la figura
plot_tree(
    model,
    feature_names=x.columns.tolist(),
    class_names=[str(c)[:20] for c in class_names],
    filled=True,   #rellena los nodos con el color de la clase  
    fontsize=8,   #tamaño de la letra
    max_depth=3   #muestra el nivel del arbol
)
plt.title("Árbol de Decisión - Predicción de INSTITUCIÓN (ChileAtiende)", fontsize=16, fontweight='bold', pad=20)
plt.savefig("arbol_decision_hd.png", dpi=600, bbox_inches="tight")
plt.close()

# --- Predicción y evaluación ---
y_pred = model.predict(X_testeo)
accuracy = accuracy_score(Y_testeo, y_pred)
print(f"\nExactitud del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")

# --- resultados de instituciones analizadas ---
print(f"\nInstituciones únicas a predecir: {len(class_names)}")
print(f"\nTop 5 Instituciones más frecuentes:")
instituciones_test = pd.Series(y).value_counts().head(5)
for inst, count in instituciones_test.items():
    print(f"  - {inst}: {count} trámites")
