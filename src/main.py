# ============================================================
# INSTALACIÓN PREVIA
# ============================================================
# uv add pandas numpy scikit-learn gradio seaborn matplotlib

# ============================================================
# SCRIPT COMPLETO
# ============================================================

import gradio as gr
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("🔄 Cargando dataset...")
# Cargar el CSV (asegúrate de que esté en la misma carpeta)
path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_dataset = os.path.join(path_root, 'dataset', 'teen_phone_addiction_dataset.csv')
df = pd.read_csv(path_dataset)

# ============================================================
# 1. LIMPIEZA Y PREPROCESAMIENTO
# ============================================================

# Eliminar columnas inútiles
columns_to_drop = ['ID', 'Name', 'Location']
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

# Variable objetivo (target)
target_col = 'Addiction_Level'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col]

# Identificar columnas categóricas
categorical_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
print(f"📊 Columnas categóricas: {categorical_cols}")

# Codificar categóricas (LabelEncoder para cada una)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    label_encoders[col] = le

# Guardar nombres de todas las características para la GUI
all_feature_names = X_raw.columns.tolist()

# Separar en principales (para el formulario) y avanzadas
main_features = [
    'Age', 'Gender', 'Daily_Usage_Hours', 'Sleep_Hours',
    'Social_Interactions', 'Anxiety_Level', 'Depression_Level',
    'Self_Esteem', 'Screen_Time_Before_Bed', 'Phone_Checks_Per_Day',
    'Time_on_Social_Media', 'Time_on_Gaming'
]
# Las columnas restantes van a la sección avanzada
advanced_features = [col for col in all_feature_names if col not in main_features]

print(f"✅ Características principales: {len(main_features)}")
print(f"✅ Características avanzadas: {len(advanced_features)}")

# ============================================================
# 2. ENTRENAMIENTO DEL MODELO (RNA)
# ============================================================

# Escalar TODOS los datos (necesario para RNA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42
)

print(f"📚 Entrenamiento: {len(X_train)} muestras, Prueba: {len(X_test)} muestras")

# Red Neuronal para REGRESIÓN (predice valor continuo)
modelo = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # 2 capas ocultas
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    random_state=42,
    verbose=True
)

print("🧠 Entrenando la Red Neuronal Artificial (Regresión)...")
modelo.fit(X_train, y_train)

# Evaluación
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ MAE: {mae:.3f}")
print(f"✅ R²: {r2:.3f}")

# ============================================================
# 3. FUNCIÓN DE PREDICCIÓN CON SECCIÓN AVANZADA
# ============================================================

# Calcular medianas para rellenar valores faltantes en sección avanzada
median_values = X_raw.median().to_dict()

def predecir_adiccion(*args):
    """
    Recibe:
    - Primero los valores de las características principales (en orden)
    - Luego los valores de las características avanzadas (en orden)
    """
    n_main = len(main_features)
    n_advanced = len(advanced_features)

    # Valores principales
    main_values = list(args[:n_main])
    # Valores avanzados (pueden ser None si no se llenaron)
    advanced_values = list(args[n_main:])

    # Rellenar valores avanzados faltantes con la mediana
    for i, val in enumerate(advanced_values):
        if val is None or np.isnan(val):
            col_name = advanced_features[i]
            advanced_values[i] = median_values[col_name]

    # Construir el vector completo (en el orden original de las columnas)
    full_values = []
    for col in all_feature_names:
        if col in main_features:
            idx = main_features.index(col)
            full_values.append(main_values[idx])
        else:
            idx = advanced_features.index(col)
            full_values.append(advanced_values[idx])

    # Convertir a array y escalar
    entrada = np.array(full_values).reshape(1, -1)
    entrada_scaled = scaler.transform(entrada)

    # Predicción
    prediccion = modelo.predict(entrada_scaled)[0]

    # Redondear a 1 decimal
    prediccion = round(prediccion, 1)

    # Interpretación
    if prediccion >= 8.0:
        nivel = "🔴 **ALTA**"
        recomendacion = "🚨 Se recomienda reducir el tiempo de pantalla y buscar ayuda profesional."
    elif prediccion >= 6.0:
        nivel = "🟡 **MODERADA**"
        recomendacion = "⚠️ Considera establecer límites de uso y aumentar actividades físicas."
    elif prediccion >= 4.0:
        nivel = "🟢 **BAJA**"
        recomendacion = "✅ Mantén hábitos saludables, pero sigue monitoreando tu uso."
    else:
        nivel = "🟣 **MUY BAJA**"
        recomendacion = "🌟 Excelente! Sigue manteniendo hábitos saludables."

    resultado = f"""
## 📱 Nivel de Adicción Predicho: **{prediccion}/10**
### Categoría: {nivel}

{recomendacion}

---
**📊 Precisión del modelo:** R² = {r2:.3f} (Error MAE = {mae:.3f})
*El modelo fue entrenado con {len(X_train)} registros de adolescentes.*
"""
    return resultado

# ============================================================
# 4. CONSTRUIR LA INTERFAZ GRADIO
# ============================================================

# --- Sección Principal (campos más relevantes) ---
main_inputs = []
for col in main_features:
    # Determinar tipo de control
    if col in categorical_cols:
        # Si es categórica, usar Radio o Dropdown
        unique_vals = sorted(df[col].unique())
        # Decodificar si estaba codificada
        if col in label_encoders:
            # Mostrar los nombres originales
            original_vals = label_encoders[col].inverse_transform(unique_vals)
            choices = list(original_vals)
        else:
            choices = [str(v) for v in unique_vals]
        main_inputs.append(
            gr.Dropdown(choices=choices, label=col, value=choices[0])
        )
    else:
        # Si es numérica, usar Slider
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        main_inputs.append(
            gr.Slider(min_val, max_val, value=mean_val, label=col, step=0.1)
        )

# --- Sección Avanzada (campos extra, opcionales) ---
advanced_inputs = []
for col in advanced_features:
    if col in categorical_cols:
        unique_vals = sorted(df[col].unique())
        if col in label_encoders:
            original_vals = label_encoders[col].inverse_transform(unique_vals)
            choices = list(original_vals)
        else:
            choices = [str(v) for v in unique_vals]
        advanced_inputs.append(
            gr.Dropdown(choices=choices, label=col, value=choices[0])
        )
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        advanced_inputs.append(
            gr.Slider(min_val, max_val, value=mean_val, label=col, step=0.1)
        )

# Combinar todos los inputs (primero principales, luego avanzados)
all_inputs = main_inputs + advanced_inputs

# Crear la interfaz con pestañas o secciones
with gr.Blocks(title="📱 Predicción de Adicción al Teléfono") as demo:
    gr.Markdown("""
    # 📱 Predicción de Adicción al Teléfono en Adolescentes
    ### Ingresa los datos del estudiante para obtener una predicción del nivel de adicción (0-10)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📋 Sección Principal")
            main_group = gr.Column()
            # Agregar los inputs principales aquí
            for inp in main_inputs:
                main_group.__enter__()
                inp.render()
                main_group.__exit__(None, None, None)

        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Sección Avanzada (Opcional)")
            gr.Markdown("*Deja vacío para usar valores promedio.*")
            advanced_group = gr.Column()
            for inp in advanced_inputs:
                advanced_group.__enter__()
                inp.render()
                advanced_group.__exit__(None, None, None)

    predict_btn = gr.Button("🔮 Predecir Adicción", variant="primary")
    output = gr.Markdown("_Esperando datos..._")

    # Conectar botón con la función
    predict_btn.click(
        fn=predecir_adiccion,
        inputs=all_inputs,
        outputs=output
    )

    gr.Markdown(f"""
    ---
    **📊 Información del Modelo**
    - **Dataset:** {len(df)} adolescentes encuestados
    - **Modelo:** MLPRegressor (2 capas ocultas: 100 y 50 neuronas)
    - **Precisión:** R² = {r2:.3f} (Error MAE = {mae:.3f})
    - **Escala de adicción:** 0 (mínimo) a 10 (máximo)
    """)

# ============================================================
# 5. EJECUTAR LA APLICACIÓN
# ============================================================

if __name__ == "__main__":
    print("🚀 Lanzando la aplicación web...")
    demo.launch(share=True)
