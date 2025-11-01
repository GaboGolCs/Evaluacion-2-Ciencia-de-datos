import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# No usamos Pipeline ni OneHotEncoder

# ... (Bloque de lectura y definición de X, y sigue igual) ...
df = pd.read_csv("nuevoCSV.csv", sep=";", encoding="utf-8", dtype=str)
df = df["TIPO_CONSULTA"]
y = df["TIPO_CONSULTA"].map({"CONSULTA": 0, "TRAMITE": 1})
X = df[["INSTITUCION", "MATERIA", "SUBMATERIA"]] 

# 1. PASO CRÍTICO: DIVIDIR PRIMERO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 2. APRENDER (FIT) SOLO DE ENTRENAMIENTO
# Aplicamos get_dummies al set de entrenamiento. Este set define las columnas.
X_train_encoded = pd.get_dummies(X_train, drop_first=True)

# 3. APLICAR (TRANSFORM) AL SET DE PRUEBA
# Usamos reindex para asegurar que X_test tenga exactamente las mismas columnas que X_train.
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Alinear las columnas (El paso manual del Pipeline)
# Mantenemos las columnas de X_train_encoded y rellenamos con 0 si faltan en X_test
columnas_train = X_train_encoded.columns 
X_test_encoded = X_test_encoded.reindex(columns=columnas_train, fill_value=0)


# 4. Entrenar el modelo (con los datos transformados)
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_encoded, y_train)

# 5. Evaluar (Usamos el set de prueba transformado para la predicción)
print("Score:", modelo.score(X_test_encoded, y_test))