import gradio as gr
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Cargar dataset
print("Cargando dataset...")
path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_dataset = os.path.join(path_root, 'dataset', 'teen_phone_addiction.csv')
df = pd.read_csv(path_dataset)

# ------------------------------------------------------------
# 1. LIMPIEZA Y PREPROCESAMIENTO
# ------------------------------------------------------------
columns_to_drop = ['ID', 'Name', 'Location']
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

target_col = 'Addiction_Level'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col]

# Identificar columnas categóricas (tipo object)
categorical_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_raw.select_dtypes(exclude=['object']).columns.tolist()

# Guardar categorías originales para los dropdowns de la interfaz
original_categories = {col: sorted(df[col].astype(str).unique()) for col in categorical_cols}

# Detectar columnas numéricas que son binarias (0/1) para usar Radio
binary_cols = []
for col in numeric_cols:
    unique_vals = set(pd.unique(df[col].dropna()))
    if unique_vals.issubset({0, 1}):
        binary_cols.append(col)
print(f"Columnas binarias detectadas: {binary_cols}")

# Lista completa de nombres de columnas (antes de codificar)
all_feature_names = X_raw.columns.tolist()

# Características principales y avanzadas (nombres originales)
main_features = [
    'Daily_Usage_Hours', 'Apps_Used_Daily', 'Time_on_Social_Media',
    'Time_on_Gaming', 'Phone_Checks_Per_Day', 'Sleep_Hours'
]
advanced_features = [col for col in all_feature_names if col not in main_features]

# ------------------------------------------------------------
# 2. PREPROCESAMIENTO CON ONE-HOT ENCODING
# ------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),                     # numéricas sin cambio
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)  # one-hot
    ]
)

# Aplicar el preprocesador y luego escalar
X_processed = preprocessor.fit_transform(X_raw)
# Convertir a array denso si es disperso
if hasattr(X_processed, 'toarray'):
    X_processed = X_processed.toarray()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 3. ENTRENAMIENTO DEL MODELO
# ------------------------------------------------------------
modelo = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    random_state=42,
    verbose=False
)

print("Entrenando la Red Neuronal...")
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

# ------------------------------------------------------------
# 4. FUNCIÓN DE PREDICCIÓN (Gradio)
# ------------------------------------------------------------
def predecir_adiccion(*args):
    n_main = len(main_features)
    main_values = list(args[:n_main])
    advanced_values = list(args[n_main:])

    # Construir un diccionario con los valores crudos en el orden de las columnas originales
    input_dict = {}
    for col in all_feature_names:
        if col in main_features:
            idx = main_features.index(col)
            val = main_values[idx]
        else:
            idx = advanced_features.index(col)
            val = advanced_values[idx]

        # Si la columna es binaria (Radio "Sí"/"No") convertir a 1/0
        if col in binary_cols:
            val = 1.0 if val == "Sí" else 0.0
        # Para las demás, mantener el valor (numérico o string de categoría)
        input_dict[col] = val

    # Crear DataFrame con una fila
    input_df = pd.DataFrame([input_dict])

    # Aplicar el preprocesador (One-Hot para categóricas) y luego el escalador
    input_processed = preprocessor.transform(input_df)
    if hasattr(input_processed, 'toarray'):
        input_processed = input_processed.toarray()
    input_scaled = scaler.transform(input_processed)

    # Predecir
    prediccion = modelo.predict(input_scaled)[0]
    prediccion = float(np.clip(round(prediccion, 1), 0, 10))

    # Interpretación
    if prediccion >= 8.0:
        nivel = "🔴 **ALTA**"
        recomendacion = "Se recomienda reducir el tiempo de pantalla y buscar ayuda profesional."
    elif prediccion >= 6.0:
        nivel = "🟡 **MODERADA**"
        recomendacion = "Considera establecer límites de uso y aumentar actividades físicas."
    elif prediccion >= 4.0:
        nivel = "🟢 **BAJA**"
        recomendacion = "Mantén hábitos saludables, pero sigue monitoreando tu uso."
    else:
        nivel = "🟣 **MUY BAJA**"
        recomendacion = "Excelente! Sigue manteniendo hábitos saludables."

    resultado = f"""
## 📱 Nivel de Adicción Predicho: **{prediccion}/10**
### Categoría: {nivel}

{recomendacion}

---
**📊 Precisión del modelo:** R² = {r2:.3f} (Error MAE = {mae:.3f})
*El modelo fue entrenado con {len(X_train)} registros de adolescentes.*
"""
    return resultado

# ------------------------------------------------------------
# 5. INTERFAZ (sin cambios en los controles)
# ------------------------------------------------------------
def crear_input(col):
    if col in categorical_cols:
        choices = original_categories[col]
        return gr.Dropdown(choices=choices, label=col, value=choices[0])
    elif col in binary_cols:
        return gr.Radio(choices=["No", "Sí"], label=col, value="No")
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        return gr.Slider(min_val, max_val, value=mean_val, label=col, step=0.1)

main_inputs = [crear_input(col) for col in main_features]
advanced_inputs = [crear_input(col) for col in advanced_features]
all_inputs = main_inputs + advanced_inputs

with gr.Blocks(title="Predicción de Adicción al Teléfono") as demo:
    gr.Markdown("""
    # 📱 Predicción de Adicción al Teléfono en Adolescentes
    ### Ingresa los datos del estudiante para obtener una predicción del nivel de adicción (0-10)
    """)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📋 Sección Principal")
            for inp in main_inputs:
                inp.render()
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Sección Avanzada (Opcional)")
            gr.Markdown("*Deja el valor por defecto si no lo conoces.*")
            for inp in advanced_inputs:
                inp.render()
    predict_btn = gr.Button("Predecir Adicción", variant="primary")
    output = gr.Markdown("_Esperando datos..._")
    predict_btn.click(fn=predecir_adiccion, inputs=all_inputs, outputs=output)
    gr.Markdown(f"""
    ---
    **📊 Información del Modelo**
    - **Dataset:** {len(df)} adolescentes encuestados
    - **Modelo:** MLPRegressor (2 capas ocultas: 100 y 50 neuronas)
    - **Precisión:** R² = {r2:.3f} (Error MAE = {mae:.3f})
    - **Escala de adicción:** 0 (mínimo) a 10 (máximo)
    """)

if __name__ == "__main__":
    print("Lanzando la aplicación web...")
    demo.launch(share=True)
