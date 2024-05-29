import streamlit as st
from streamlit_option_menu import option_menu

import about
import account
import cancerdata
import feedbc
import home
import news
import predict
import predictdata

st.set_page_config(
    page_title="Breast Care",
)

class MultiApp:
    
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func,
        })

    def run():
        with st.sidebar:
            st.image('image/logo-removebg.png', width=200, caption='')
            app = option_menu(
                menu_title='Breast Care ',
                options=['Home','News','Predict Data','Predict','Cancer Data','Feedback','Account','about'],
                icons=['house-fill','newspaper','book','graph-up','database','chat-text-fill','person-circle','info-circle-fill'],
                # menu_icon= st.image('img/breast-cancer.png', width=100),
                default_index=1,
                styles={
                    "container": {"padding": "5!important","background": 'linear-gradient(to right, rgba(67, 173, 234, 0.8), rgba(221, 122, 229, 0.8))'},
                    "icon": {"color": "white", "font-size": "23px"}, 
                    "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
                    "nav-link-selected": {"background": 'linear-gradient(to right, rgba(67, 173, 234, 0.8), rgba(221, 122, 229, 0.8))'}
                }
                
            )

        
        if app == "Home":
            home.app()
        if app=='News':
            news.app()
        if app=='Predict':
            predict.app()
        if app=='Predict Data':
            predictdata.app()
        if app == "Cancer Data":
            cancerdata.app()
        if app == 'Feedback':
            feedbc.app()
        if app == "Account":
            account.app()
        if app == 'about':
            about.app()
                
    run() 
