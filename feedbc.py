import streamlit as st
from firebase_admin import firestore


def app():
    st.title(':violet[Kritik] & :violet[Saran]  :sunglasses:')
    
    if 'db' not in st.session_state:
        st.session_state.db = ''

    db=firestore.client()
    st.session_state.db=db
    
    ph = ''
    if st.session_state.username=='':
        ph = 'Silakan Login terlebih dahulu!!'
    else:
        ph='Masukkan Pesan anda'    
    send=st.text_area(label=' :orange[+ Kritik dan Saran]',placeholder=ph,height=None, max_chars=500)
    if st.button('Kirim',use_container_width=20):
        if send!='':
                    
            info = db.collection('Send').document(st.session_state.username).get()
            if info.exists:
                info = info.to_dict()
                if 'Content' in info.keys():
                
                    pos=db.collection('Send').document(st.session_state.username)
                    pos.update({u'Content': firestore.ArrayUnion([u'{}'.format(send)])})
                else:
                    
                    data={"Content":[send],'Username':st.session_state.username}
                    db.collection('Send').document(st.session_state.username).set(data)    
            else:
                    
                data={"Content":[send],'Username':st.session_state.username}
                db.collection('Send').document(st.session_state.username).set(data)
                
            st.success('Horee Terkirim, Terima Kasih...')
    
    st.header(' :violet[Latest Messeges] ')
    
    docs = db.collection('Send').get()
            
    for doc in docs:
        d=doc.to_dict()
        try:
            st.markdown("""
                <style>
                .stTextArea [data-baseweb=base-input] [disabled=""]{
                    background: linear-gradient(to right, rgba(67, 173, 234, 0.8), rgba(221, 122, 229, 0.8));
                    -webkit-text-fill-color: white;
                }
                </style>
                """,unsafe_allow_html=True)
            
            st.text_area(label=':green[Feedback by:] '+':orange[{}]'.format(d['Username']), value=d['Content'][-1], height=20, disabled=True)
            
        except:
            if st.session_state.username=='':
                st.text('Please Login first')