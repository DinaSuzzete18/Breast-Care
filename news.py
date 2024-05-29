url ='https://www.who.int/news-room/fact-sheets/detail/breast-cancer#:~:text=Breast%20cancer%20is%20a%20disease,producing%20lobules%20of%20the%20breast.'

import streamlit as st


def app():
    st.subheader('Ada Berita Apa nih?')
    st.markdown(f'<a href="{url}" target="_blank" style="background-color: #008CBA; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 4px;">Detail</a>', unsafe_allow_html=True)

    st.subheader('Apa itu Kanker Payudara?')
    st.video("https://youtu.be/fYTgVO-D3f8?si=7eK2tKMLj9qiPdpV")
    st.markdown('Kanker payudara adalah tumor ganas yang tumbuh di dalam jaringan payudara. Kanker bisa mulai tumbuh di dalam kelenjar susu, saluran susu, jaringan lemak maupun jaringan ikat pada payudara.Di Indonesia, kanker payudara menjadi salah satu penyakit penyumbang kematian pertama yang disebabkan kanker serta jumlah pasiennya yang menempati urutan pertama terbanyak.Pada tahun 2020, data dari Globocan yang dikutip dari Kementerian Kesehatan, mencatat jumlah kasus baru kanker payudara mencapai 68.858 kasus, atau sekitar 16,6% dari total 396.914 kasus baru kanker di Indonesia. Adapun jumlah kematiannya mencapai 22 ribu lebih kasus.  ')

    st.subheader('Bagaimana gelaja kanker payudara?')
    st.image('image/gejala.png')
    st.markdown ('Gejala kanker payudara pada setiap orang bisa berbeda. Tetapi gejala yang paling umum adanya benjolan di payudara atau ketiak. Gejala tersebut umumnya terdeteksi ketika melakukan pemeriksaan payudara rutin, baik pemeriksaan sendiri (SADARI), maupun pemeriksaan payudara klinis (SADANIS). Berikut gejala kanker payudara yang sebaiknya di warpadai antara lain: ')
    st.markdown ('1. Muncul sebuah benjolan yang terasa dari jaringan payudara di sekitarnya') 
    st.markdown ('2. Saat didorong oleh jari tangan, benjolan bisa digerakkan dengan mudah di bawah kulit')
    st.markdown ('3. Benjolan atau massa juga bisa muncul di ketiak, sekitar tulang selangka atau di bawah lengan')
    st.markdown ('4. Perubahan ukuran atau bentuk payudara')
    st.markdown ('5. Keluar cairan yang abnormal dari puting susu. Cairan juga mengandung darah, berwarna kuning sampai hijau, atau bisa bernanah')
    st.markdown ('6. Warna atau tekstur kulit payudara, puting susu maupun daerah berwarna coklat tua disekililing puting susu(areola) mengalami perubahan')
    st.markdown ('7. Payudara tampak kemerahan')
    st.markdown ('8. Kulit sekitar puting bersisik') 
    st.markdown ('9. Puting susu tera gatal atau tertarik ke dalam')
    st.markdown ('10.Nyeri payudara atau pembengkakan salah satu payudara')

    st.markdown('Memiliki satu atau lebih dari gejala tersebut tidak selalu berarti menderita kanker payudara. Pasalnya, cairan dari puting juga bisa jadi dipicu oleh infeksi. Untuk itulah, menemui dokter supaya mendapatkan penanganan yang tepat merupakan langkah yang tepat.')