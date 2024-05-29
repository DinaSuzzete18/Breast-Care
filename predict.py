import pickle

import numpy as np
import streamlit as st

def app():
    # Load the trained model
    with open('model.sav', 'rb') as model_file:
        breastcancer_model = pickle.load(model_file)

    st.title('Lets Predict your :violet [breast cancer]')
    # ini harus pake float ga sihh bingunggg
    radius_mean = st.text_input('input radius mean')
    texture_mean = st.text_input('input texture mean')
    perimeter_mean = st.text_input('input perimeter mean')
    area_mean = st.text_input('input area mean')
    smoothness_mean = st.text_input('input smoothness mean')
    compactness_mean = st.text_input('input compactness mean')
    concavity_mean = st.text_input('input concavity mean')
    concave_points_mean = st.text_input('input concave points mean')
    symmetry_mean = st.text_input('input symmetry mean')
    fractal_dimension_mean = st.text_input('input fractal dimension mean')
    radius_se = st.text_input('input radius se')
    texture_se = st.text_input('input texture se')
    perimeter_se = st.text_input('input perimeter se')
    area_se = st.text_input('input area se')
    smoothness_se = st.text_input('input smoothness se')
    compactness_se = st.text_input('input compactness se')
    concavity_se = st.text_input('input concavity se')
    concave_points_se = st.text_input('input concave points se')
    symmetry_se = st.text_input('input symmetry se')
    fractal_dimension_se = st.text_input('input fractal dimension se')
    radius_worst = st.text_input('input radius worst')
    texture_worst = st.text_input('input texture worst')
    perimeter_worst = st.text_input('input perimeter worst')
    area_worst = st.text_input('input area worst')
    smoothness_worst = st.text_input('input smothness worst')
    compactness_worst = st.text_input('input compactness worst')
    concavity_worst = st.text_input('input concavity worst')
    concave_points_worst = st.text_input('input concave points worst')
    symmetry_worst = st.text_input('input symmetry worst')
    fractal_dimension_worst = st.text_input('input fractal dimension worst')

    # code prediksi
    cancer_diagnosis = ''

    # tombol prediksi
    if st.button('Predict'):
        # cancer_predict = breastcancer_model.predict([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se , concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst])
        
        # Validasi input tidak boleh kosong
        if (radius_mean != '' and texture_mean != '' and perimeter_mean != '' and area_mean != '' and smoothness_mean != '' and compactness_mean != '' and concavity_mean != '' and concave_points_mean != '' and symmetry_mean != '' and fractal_dimension_mean != '' and radius_se != '' and texture_se != '' and perimeter_se != '' and area_se != '' and smoothness_se != '' and compactness_se != '' and concavity_se != '' and concave_points_se != '' and symmetry_se != '' and fractal_dimension_se != '' and radius_worst != '' and texture_worst != '' and perimeter_worst != '' and area_worst != '' and smoothness_worst != '' and compactness_worst != '' and concavity_worst != '' and concave_points_worst != '' and symmetry_worst != '' and fractal_dimension_worst != ''):
            input_data = np.array([float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean), float(smoothness_mean), float(compactness_mean), float(concavity_mean), float(concave_points_mean), float(symmetry_mean), float(fractal_dimension_mean), float(radius_se), float(texture_se), float(perimeter_se), float(area_se), float(smoothness_se), float(compactness_se), float(concavity_se), float(concave_points_se), float(symmetry_se), float(fractal_dimension_se), float(radius_worst), float(texture_worst), float(perimeter_worst), float(area_worst), float(smoothness_worst), float(compactness_worst), float(concavity_worst), float(concave_points_worst), float(symmetry_worst), float(fractal_dimension_worst)]).reshape(1, -1)
            cancer_predict = breastcancer_model.predict(input_data)
            
            if(cancer_predict[0]=='1'):
                cancer_diagnosis = 'Pasien memilki Tumor Ganas'
            else:
                cancer_diagnosis = 'Pasien memiliki Tumor Jinak'
                
            st.success(cancer_diagnosis)
        else:
            st.error('Harap isi semua kolom input')
        