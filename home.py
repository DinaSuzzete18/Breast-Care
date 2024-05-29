import streamlit as st
from firebase_admin import firestore


def app():
    st.title('Welcome to :violet[Breast Care] :sunglasses:')
    
    #subheader 
    st.subheader("Deteksi Dini Kanker Payudara dari Sekarang")
    # text 
    st.markdown('Kanker payudara merupakan penyakit terbanyak pada wanita dan angka kematiannya menempati posisi kedua pada kasus kanker. Kematian tersebut dapat dikurangi dengan melakukan deteksi dini terhadap sel kanker. Prediksi kanker ini dapat membantu pasien untuk berkonsultasi dengan dokter lebih cepat.')
    st.image('image/head.jpg', use_column_width=True)
    