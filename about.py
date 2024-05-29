import streamlit as st
from PIL import Image

def app():
    st.title('Breast Care')
    st.markdown('adalah sebuah website untuk memprediksi apakah kanker tersebut jinak atau ganas,dengan disertai informasi mengenai kanker payudara.')
    st.text('Created by: Kelompok 12 Project Capstone - MSIB Batch 6')
    st.markdown("<h4 style='text-align: center;'>Anggota Kelompok</h4>", unsafe_allow_html=True)

    Rifqi = Image.open('image/Rifqi.jpg')
    Dina = Image.open('image/dina.jpg')
    Laela = Image.open('image/laela.jpg')
    Novia = Image.open('image/novia.jpg')
    Daffin = Image.open('image/dafin.jpg')
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center;'>Rifqi Nurul Qomariah</h3>", unsafe_allow_html=True)
        st.image(Rifqi, caption='Manajemen Informatika, Politeknik Negeri Jember')
    with col2: 
        st.markdown("<h3 style='text-align: center;'>Dina Suzzete Sitorus</h3>", unsafe_allow_html=True)
        st.image(Dina, caption='Matematika, Universitas Sriwijaya')
    with col1:
        st.markdown("<h3 style='text-align: center;'>Laelatusya’diyah</h3>", unsafe_allow_html=True)
        st.image(Laela, caption='ilmu komputer, Universitas Pendidikan Indonesia')
    with col2: 
        st.markdown("<h3 style='text-align: center;'>Novia Citra Andini</h3>", unsafe_allow_html=True)
        st.image(Novia, caption='Matematika, Universitas Mataram')
    st.markdown("<h3 style='text-align: center;'>Muhammad Daffin Salman M.</h3>", unsafe_allow_html=True)
    st.image(Daffin, caption='Agribisnis, Universitas Padjadjaran')
    
    