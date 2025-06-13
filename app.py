import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsão de Obesidade", layout="wide")

# --- CARREGAMENTO DOS ARQUIVOS DO MODELO ---
@st.cache_resource
def load_model_assets():
    """Carrega o modelo, o scaler, o label encoder e as colunas do modelo."""
    model = joblib.load('modelo_final_lgbm.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    model_columns = joblib.load('colunas_modelo.joblib')
    return model, scaler, label_encoder, model_columns

model, scaler, label_encoder, model_columns = load_model_assets()


# --- INTERFACE DO USUÁRIO (com ajustes nos sliders) ---
st.title('Sistema Preditivo para Níveis de Obesidade')
st.write('Esta aplicação utiliza um modelo de Machine Learning (LightGBM) para prever o nível de obesidade de um indivíduo com base em seus hábitos e características físicas.')

col1, col2, col3 = st.columns(3)

# --- Listas de opções para os seletores e sliders ---
opcoes_sim_nao = ['Sim', 'Não']
opcoes_genero = ['Masculino', 'Feminino']
opcoes_caec = ['Não', 'Às vezes', 'Frequentemente', 'Sempre']
opcoes_calc = ['Não', 'Às vezes', 'Frequentemente']
opcoes_mtrans = ['Transporte Público', 'Automóvel', 'Andando', 'Motocicleta', 'Bicicleta']

# NOVAS OPÇÕES PARA OS SLIDERS DE TEXTO
opcoes_fcvc = ['Nunca ou Raramente', 'Às vezes', 'Sempre']
opcoes_ncp = ['1', '2', '3', '4'] # NCP é numérico, mas podemos usar select_slider
opcoes_ch2o = ['Menos de 1 Litro', 'Entre 1 e 2 Litros', 'Mais de 2 Litros']
opcoes_faf = ['Nenhuma', '1 ou 2 dias', '2 a 4 dias', '4 ou 5 dias']
opcoes_tue = ['0 a 2 horas', '3 a 5 horas', 'Mais de 5 horas']


with col1:
    st.header("Características Físicas")
    gender = st.selectbox('Gênero', opcoes_genero)
    age = st.number_input('Idade', min_value=1, max_value=100, value=25)
    height = st.number_input('Altura (metros)', min_value=0.5, max_value=2.5, value=1.70, format="%.2f")
    weight = st.number_input('Peso (kg)', min_value=10.0, max_value=200.0, value=70.0, format="%.1f")

with col2:
    st.header("Hábitos Alimentares")
    family_history = st.selectbox('Histórico Familiar de Sobrepeso?', opcoes_sim_nao)
    favc = st.selectbox('Consome alimentos de alta caloria frequentemente (FAVC)?', opcoes_sim_nao)
    # MUDANÇA: trocando st.slider por st.select_slider
    fcvc = st.select_slider('Frequência de consumo de vegetais (FCVC)', options=opcoes_fcvc, value='Às vezes')
    ncp = st.select_slider('Número de refeições principais (NCP)', options=opcoes_ncp, value='3')
    caec = st.selectbox('Consome algo entre as refeições (CAEC)?', opcoes_caec)
    scc = st.selectbox('Monitora o consumo de calorias (SCC)?', opcoes_sim_nao)
    
with col3:
    st.header("Outros Hábitos")
    smoke = st.selectbox('Fuma?', opcoes_sim_nao)
    # MUDANÇA: trocando st.slider por st.select_slider
    ch2o = st.select_slider('Consumo diário de água (CH2O)', options=opcoes_ch2o, value='Entre 1 e 2 Litros')
    faf = st.select_slider('Frequência de atividade física (FAF)', options=opcoes_faf, value='1 ou 2 dias')
    tue = st.select_slider('Tempo de uso de dispositivos (TUE)', options=opcoes_tue, value='0 a 2 horas')
    calc = st.selectbox('Consumo de álcool (CALC)?', opcoes_calc)
    mtrans = st.selectbox('Meio de transporte principal (MTRANS)', opcoes_mtrans)

# Botão de previsão
if st.button('**Prever Nível de Obesidade**', use_container_width=True):
    
    # --- MAPEAMENTO E PREPARAÇÃO DOS DADOS DE ENTRADA ---
    # 1. Dicionários de mapeamento para as entradas em texto
    map_sim_nao = {'Sim': 'yes', 'Não': 'no'}
    map_genero = {'Masculino': 'Male', 'Feminino': 'Female'}
    map_caec = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
    map_calc = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently'}
    map_mtrans = {
        'Transporte Público': 'Public_Transportation', 'Automóvel': 'Automobile', 
        'Andando': 'Walking', 'Motocicleta': 'Motorbike', 'Bicicleta': 'Bike'
    }
    # NOVOS MAPS para os sliders de texto
    map_fcvc = {'Nunca ou Raramente': 1.0, 'Às vezes': 2.0, 'Sempre': 3.0}
    map_ch2o = {'Menos de 1 Litro': 1.0, 'Entre 1 e 2 Litros': 2.0, 'Mais de 2 Litros': 3.0}
    map_faf = {'Nenhuma': 0.0, '1 ou 2 dias': 1.0, '2 a 4 dias': 2.0, '4 ou 5 dias': 3.0}
    map_tue = {'0 a 2 horas': 0.0, '3 a 5 horas': 1.0, 'Mais de 5 horas': 2.0}

    # 2. Criar um DataFrame com os dados de entrada, já mapeando para os valores numéricos/inglês
    input_data = {
        'Gender': map_genero[gender], 'Age': age, 'Height': height, 'Weight': weight,
        'family_history': map_sim_nao[family_history], 'FAVC': map_sim_nao[favc], 
        'FCVC': map_fcvc[fcvc], # Mapeando o texto para número
        'NCP': float(ncp), # Convertendo o texto '3' para número 3.0
        'CAEC': map_caec[caec], 'SMOKE': map_sim_nao[smoke], 
        'CH2O': map_ch2o[ch2o], # Mapeando o texto para número
        'SCC': map_sim_nao[scc], 
        'FAF': map_faf[faf], # Mapeando o texto para número
        'TUE': map_tue[tue], # Mapeando o texto para número
        'CALC': map_calc[calc], 
        'MTRANS': map_mtrans[mtrans]
    }
    input_df = pd.DataFrame([input_data])
    
    # 3. Aplicar as mesmas transformações numéricas do treinamento
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['family_history'] = input_df['family_history'].map({'yes': 1, 'no': 0})
    input_df['FAVC'] = input_df['FAVC'].map({'yes': 1, 'no': 0})
    input_df['SCC'] = input_df['SCC'].map({'yes': 1, 'no': 0})
    input_df['SMOKE'] = input_df['SMOKE'].map({'yes': 1, 'no': 0})
    
    caec_map_num = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_map_num = {'no': 0, 'Sometimes': 1, 'Frequently': 2}
    input_df['CAEC'] = input_df['CAEC'].map(caec_map_num)
    input_df['CALC'] = input_df['CALC'].map(calc_map_num)

    input_df = pd.get_dummies(input_df, columns=['MTRANS'], dtype=int)
    
    # 4. Alinhar colunas com as do modelo
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]

    # 5. Padronizar os dados
    input_scaled = scaler.transform(input_df)
    
    # --- PREDIÇÃO E EXIBIÇÃO DO RESULTADO ---
    prediction_numeric = model.predict(input_scaled)
    prediction_label_english = label_encoder.inverse_transform(prediction_numeric)
    
    map_resultado = {
        'Normal_Weight': 'Peso Normal',
        'Overweight_Level_I': 'Sobrepeso Nível I',
        'Overweight_Level_II': 'Sobrepeso Nível II',
        'Obesity_Type_I': 'Obesidade Tipo I',
        'Obesity_Type_II': 'Obesidade Tipo II',
        'Obesity_Type_III': 'Obesidade Tipo III',
        'Insufficient_Weight': 'Peso Insuficiente'
    }
    
    resultado_final_pt = map_resultado.get(prediction_label_english[0], "Resultado não encontrado")
    
    st.success(f"**Resultado da Predição:** O nível de obesidade previsto é **{resultado_final_pt}**.")
