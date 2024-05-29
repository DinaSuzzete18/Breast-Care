import streamlit as st 

def app():
    st.header(":mailbox: Get In Touch With Me!")

contact_form = """
<form action="https://formsubmit.co/suzzetecelisitorus22@gmail.com" method="POST">
     <input type="text" name="name" required>
     <input type="email" name="email" required>
     <button type="submit">Send</button>
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)

