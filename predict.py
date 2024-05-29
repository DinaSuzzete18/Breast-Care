import streamlit as st
import pickle
import numpy as np

def load_model_and_scaler():
    # Load the trained model
    with open('model.sav', 'rb') as model_file:
        breastcancer_model = pickle.load(model_file)
    # Load the fitted scaler
    with open('scaler.sav', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return breastcancer_model, scaler

def make_prediction(input_data, model, scaler):
    input_data_as_numpy_array = np.array(input_data)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshape)
    prediction = model.predict(std_data)
    return prediction

def app():
    # Load the model and scaler
    breastcancer_model, scaler = load_model_and_scaler()

    st.title('Let\'s Predict Your Breast Cancer Diagnosis')

    # Input fields
    input_fields = {
        'Radius Mean': '',
        'Texture Mean': '',
        'Perimeter Mean': '',
        'Area Mean': '',
        'Smoothness Mean': '',
        'Compactness Mean': '',
        'Concavity Mean': '',
        'Concave Points Mean': '',
        'Symmetry Mean': '',
        'Fractal Dimension Mean': '',
        'Radius SE': '',
        'Texture SE': '',
        'Perimeter SE': '',
        'Area SE': '',
        'Smoothness SE': '',
        'Compactness SE': '',
        'Concavity SE': '',
        'Concave Points SE': '',
        'Symmetry SE': '',
        'Fractal Dimension SE': '',
        'Radius Worst': '',
        'Texture Worst': '',
        'Perimeter Worst': '',
        'Area Worst': '',
        'Smoothness Worst': '',
        'Compactness Worst': '',
        'Concavity Worst': '',
        'Concave Points Worst': '',
        'Symmetry Worst': '',
        'Fractal Dimension Worst': ''
    }

    for key in input_fields.keys():
        input_fields[key] = st.text_input(f'Input {key.lower()}')

    # Button for prediction
    if st.button('Predict'):
        # Check if all fields are filled
        if all(value != '' for value in input_fields.values()):
            input_data = [float(value) for value in input_fields.values()]
            prediction = make_prediction(input_data, breastcancer_model, scaler)
            
            if prediction[0] == 0:
                cancer_diagnosis = 'Pasien memiliki Tumor Jinak'
            else:
                cancer_diagnosis = 'Pasien memiliki Tumor Ganas'
            
            st.success(cancer_diagnosis)
        else:
            st.error('Harap isi semua kolom input')
