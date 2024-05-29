import streamlit as st
import pandas as pd

def app():
    st.subheader('Data Kanker Payudara')

    df = pd.df = pd.read_csv("data.csv")
    st.dataframe(df,height=1500, width= 5000)
