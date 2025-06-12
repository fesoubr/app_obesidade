import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Previsão de Obesidade", layout="wide")

# --- Carregamento dos artefatos do modelo ---
# O decorador @st.cache_resource vem DEPOIS do set_page_config
@st.cache_resource
def load_model_assets():
    """Carrega o modelo, o scaler, o label encoder e as colunas do modelo."""
    model = joblib.load('modelo_final_lgbm.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    model_columns = joblib.load('colunas_modelo.joblib')
    return model, scaler, label_encoder, model_columns

model, scaler, label_encoder, model_columns = load_model_assets()


# --- Interface do Usuário (UI) com Streamlit ---

st.title('Sistema Preditivo para Níveis de Obesidade')
st.write('Esta aplicação utiliza um modelo de Machine Learning (LightGBM) para prever o nível de obesidade de um indivíduo com base em seus hábitos e características físicas.')

# Organizando a UI em colunas
col1, col2, col3 = st.columns(3)


with col1:
    st.header("Características Físicas")
    gender = st.selectbox('Gênero', ['Male', 'Female'])
    age = st.number_input('Idade', min_value=1, max_value=100, value=25)
    height = st.number_input('Altura (metros)', min_value=0.5, max_value=2.5, value=1.70, format="%.2f")
    weight = st.number_input('Peso (kg)', min_value=10.0, max_value=200.0, value=70.0, format="%.1f")

with col2:
    st.header("Hábitos Alimentares")
    family_history = st.selectbox('Histórico Familiar de Sobrepeso?', ['yes', 'no'])
    favc = st.selectbox('Consome alimentos de alta caloria frequentemente (FAVC)?', ['yes', 'no'])
    fcvc = st.slider('Frequência de consumo de vegetais (FCVC)', 1.0, 3.0, 2.0, step=0.5)
    ncp = st.slider('Número de refeições principais (NCP)', 1.0, 4.0, 3.0, step=0.5)
    caec = st.selectbox('Consome algo entre as refeições (CAEC)?', ['no', 'Sometimes', 'Frequently', 'Always'])
    scc = st.selectbox('Monitora o consumo de calorias (SCC)?', ['yes', 'no'])
    
with col3:
    st.header("Outros Hábitos")
    smoke = st.selectbox('Fuma (SMOKE)?', ['yes', 'no'])
    ch2o = st.slider('Consumo diário de água (CH2O - litros)', 1.0, 3.0, 2.0, step=0.5)
    faf = st.slider('Frequência de atividade física (FAF - dias/semana)', 0.0, 3.0, 1.0, step=0.5)
    tue = st.slider('Tempo de uso de dispositivos tecnológicos (TUE - horas/dia)', 0.0, 2.0, 1.0, step=0.5)
    calc = st.selectbox('Consumo de álcool (CALC)?', ['no', 'Sometimes', 'Frequently'])
    mtrans = st.selectbox('Meio de transporte principal (MTRANS)', ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])

if st.button('**Prever Nível de Obesidade**', use_container_width=True):
    
    input_data = {
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'family_history': family_history, 'FAVC': favc, 'FCVC': fcvc,
        'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc,
        'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
    }
    
    input_df = pd.DataFrame([input_data])
    
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['family_history'] = input_df['family_history'].map({'yes': 1, 'no': 0})
    input_df['FAVC'] = input_df['FAVC'].map({'yes': 1, 'no': 0})
    input_df['SCC'] = input_df['SCC'].map({'yes': 1, 'no': 0})
    input_df['SMOKE'] = input_df['SMOKE'].map({'yes': 1, 'no': 0})
    
    caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2}
    input_df['CAEC'] = input_df['CAEC'].map(caec_map)
    input_df['CALC'] = input_df['CALC'].map(calc_map)

    input_df = pd.get_dummies(input_df, columns=['MTRANS'], dtype=int)
    
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[model_columns]

    input_scaled = scaler.transform(input_df)
    
    prediction_numeric = model.predict(input_scaled)
    
    prediction_label = label_encoder.inverse_transform(prediction_numeric)
    
    st.success(f"**Resultado da Predição:** O nível de obesidade previsto é **{prediction_label[0]}**.")