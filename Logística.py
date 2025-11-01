import pandas as pd
from sklearn.model_selection import train_test_split
# Tranformaciones de y, codificaciones, escalado
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
#Modelo y metricas de evaluación
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Leer el archivo CSV
try:
    df = pd.read_csv('nuevoCSV.csv', sep=';')
    print("DataFrame cargado exitosamente.")
except FileNotFoundError:
    print("Error: El archivo no fue encontrado.")

#Variable a adivinar
y = df['TIPO_CONSULTA']

#Pistas para adivinar
X = df[['SUCURSAL', 'INSTITUCION', 'MATERIA', 'SUBMATERIA']]

#Creamos instacia del codificador y hacemos la transformación
codificadora_binaria = LabelEncoder()
y_codificada = codificadora_binaria.fit_transform(y)

#Imprime el numero que le correspone a consulta y tramite
for i, opciones in enumerate(codificadora_binaria.classes_):
    print(f"{opciones} -> {i}")

#Creamos una variable con las columnas a codificar
variables = ['SUCURSAL', 'INSTITUCION', 'MATERIA', 'SUBMATERIA']

# Configuramos el ColumnTransformer con OneHotEncoder
#ColumnTransformer nos permite aplicar diferentes transformaciones a diferentes columnas
preprocesado = ColumnTransformer(
    transformers=[
        #nombre que que le damos al paso, el codificador que vamos a usar, y las variables a las que se les aplica
        ('binario', OneHotEncoder(handle_unknown='ignore'), variables)
    ],
    remainder='passthrough'
)

#Aplicamos el preprocesado a X
X_codificada = preprocesado.fit_transform(X)

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_codificada, 
    y_codificada, 
    test_size=0.3, 
    random_state=42, 
    #aseguramos que la proporción de clases se mantenga en ambos conjuntos para evitar solo una variante en el conjunto de prueba
    stratify=y_codificada
)


# StandardScaler  se asegura que todas las variables tengas el mismo peso en el modelo
escalador = StandardScaler(with_mean=False)
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)



#usamos max_iter para asegurarnos que el modelo converge
modelo = LogisticRegression(random_state=42, max_iter=1000)

#entrenamos el modelo
modelo.fit(X_train_escalado, y_train)

#Evaluamos el modelo
y_pred = modelo.predict(X_test_escalado)


#Porcentaje de acierto
print(f"Accuracy (Exactitud): {accuracy_score(y_test, y_pred):.2f}")

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=codificadora_binaria.classes_))
