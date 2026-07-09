# ============================================================
# INSTALACIÓN PREVIA
# ============================================================
# pip install pandas numpy scikit-learn gradio

# ============================================================
# SCRIPT COMPLETO CORREGIDO
# ============================================================

import gradio as gr
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

print("🔄 Cargando dataset...")
path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_dataset = os.path.join(path_root, 'dataset', 'teen_phone_addiction.csv')
df = pd.read_csv(path_dataset)

# ============================================================
# 1. LIMPIEZA Y PREPROCESAMIENTO
# ============================================================

columns_to_drop = ['ID', 'Name', 'Location']
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

target_col = 'Addiction_Level'
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col]

categorical_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
original_categories = {}  # Guardar valores originales

for col in categorical_cols:
    original_categories[col] = sorted(df[col].astype(str).unique())
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    label_encoders[col] = le

all_feature_names = X_raw.columns.tolist()

# Características principales
main_features = [
    'Daily_Usage_Hours',
    'Apps_Used_Daily',
    'Time_on_Social_Media',
    'Time_on_Gaming',
    'Phone_Checks_Per_Day',
    'Sleep_Hours'
]
advanced_features = [col for col in all_feature_names if col not in main_features]

print(f"✅ Características principales: {len(main_features)}")
print(f"✅ Características avanzadas: {len(advanced_features)}")

# ============================================================
# 2. ENTRENAMIENTO DEL MODELO
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.2, random_state=42
)

modelo = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    random_state=42,
    verbose=False
)

print("🧠 Entrenando la Red Neuronal...")
modelo.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ MAE: {mae:.3f}")
print(f"✅ R²: {r2:.3f}")

# ============================================================
# 3. FUNCIÓN DE PREDICCIÓN
# ============================================================

def predecir_adiccion(*args):
    n_main = len(main_features)
    n_advanced = len(advanced_features)

    main_values = list(args[:n_main])
    advanced_values = list(args[n_main:])

    full_values = []
    for col in all_feature_names:
        if col in main_features:
            idx = main_features.index(col)
            val = main_values[idx]
        else:
            idx = advanced_features.index(col)
            val = advanced_values[idx]

        if col in categorical_cols:
            # Codificar el string
            encoded = label_encoders[col].transform([str(val)])[0]
            full_values.append(encoded)
        else:
            full_values.append(float(val))

    entrada = np.array(full_values).reshape(1, -1)
    entrada_scaled = scaler.transform(entrada)
    prediccion = modelo.predict(entrada_scaled)[0]
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
# 4. INTERFAZ GRADIO
# ============================================================

main_inputs = []
for col in main_features:
    if col in categorical_cols:
        choices = original_categories[col]
        main_inputs.append(gr.Dropdown(choices=choices, label=col, value=choices[0]))
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        main_inputs.append(gr.Slider(min_val, max_val, value=mean_val, label=col, step=0.1))

advanced_inputs = []
for col in advanced_features:
    if col in categorical_cols:
        choices = original_categories[col]
        advanced_inputs.append(gr.Dropdown(choices=choices, label=col, value=choices[0]))
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        advanced_inputs.append(gr.Slider(min_val, max_val, value=mean_val, label=col, step=0.1))

all_inputs = main_inputs + advanced_inputs

with gr.Blocks(title="📱 Predicción de Adicción al Teléfono") as demo:
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

    predict_btn = gr.Button("🔮 Predecir Adicción", variant="primary")
    output = gr.Markdown("_Esperando datos..._")

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
# 5. EJECUTAR
# ============================================================

if __name__ == "__main__":
    print("🚀 Lanzando la aplicación web...")
    demo.launch(share=True)
