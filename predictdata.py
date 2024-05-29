url ='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7349542/'

import streamlit as st

def app():
    st.header('Bagaimana cara melakukan pemeriksaan Kanker Payudara?')
    st.image('image/sadanis.jpg')
    st.markdown('Kanker payudara sulit terdeteksi di tahap awal karena memiliki ukuran kecil. Pasien baru merasakan gejala ketika benjolan sudah terba karena ukurannya sudah cukup membesar. Oleh karena itu, pemeriksaan penting dilakukan berguna memastikan kanker payudara. Diusahakan untuk melakukan diagnosis dini karena kanker payudara lebih mudah diobati dan bisa disembuhkan jika masih pada stadium dini. Untuk mengetahui tanda-tanda kanker payudara, diperlukan SADANIS, atau USG Payudara rutin dilakukan. Berikut parameter pemeriksaan data kanker.   ')
    st.header('Apa Saja parameter data kanker?')
    st.markdown(f'<a href="{url}" target="_blank" style="background-color: #008CBA; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 4px;">Detail data</a>', unsafe_allow_html=True)
    st.markdown('Himpunan data kanker berisi 32 parameter. Semua parameter dapat berguna untuk mengklasifikasikan kanker; Jika parameter ini memiliki nilai yang relatif besar, itu bisa menjadi tanda jaringan ganas. Berikut sekilas penjelasan mengenai parameter yang digunakan dalam prediksi yaitu:  ')
   
   
    st.markdown ('1.  ID merupakan angka untuk identifikasi')
    st.markdown ('2.  Diagnosis merupakan diagnosis jenis kanker ganas dan jinak. Untuk jenis kanker yang berbeda, perlu untuk menentukan diagnosis jaringan yang benar jika kedua membran memiliki perawatan yang berbeda. Setelah keduanya, rata-rata estimasi, kesalahan standar, dan rata-rata radius menunjukkan rentang antara pusat dan titik pada perimeter.')
    st.markdown ('3.  Radius Mean merupakan perkiraan kesalahan standar. Radius terburuk memiliki nilai pusat tertinggi untuk kisaran perkiraan. Sangat penting untuk mengetahui jarak antara pusat dan titik karena operasi tergantung pada ukuran. Tidak ada kesempatan untuk melakukan operasi dengan tumor besar.')
    st.markdown ('4.  Texture Mean mewakili simpangan baku dari nilai skala abu-abu. Tekstur se mewakili kesalahan standar simpangan baku yang dihitung untuk nilai skala abu-abu. Tekstur se mewakili kesalahan standar simpangan baku yang dihitung untuk nilai skala abu-abu. Nilai rata-rata tertinggi dari standar deviasi untuk nilai skala abu-abu ditampilkan sebagai tekstur terburuk. Skala abu-abu biasanya digunakan untuk menemukan lokasi tumor, dan standar deviasi sangat penting untuk menemukan variasi data dan untuk menjelaskan bagaimana menyebarkan angka-angka.')
    st.markdown ('5.  Perimeter Mean erimeter mean mewakili nilai mean untuk tumor inti, sedangkan kesalahan standar mean mewakili tumor inti yang digambarkan sebagai perimeter se. Nilai tertinggi dari tumor inti ditulis pada kolom terburuk perimeter. ')
    st.markdown ('6.  Area Mean,area se, dan area titik terburuk pada nilai yang sama terkait dengan rata-rata area sel kanker, seperti yang dijelaskan sebelumnya.')
    st.markdown ('7.  Smoothess Mean adalah rata-rata untuk variasi regional dalam rentang radius, kehalusan se mewakili kesalahan standar rata-rata variasi lokal dalam panjang radius, dan nilai rata-rata terbesar ditampilkan sebagai kehalusan terburuk.')
    st.markdown ('8.  Compactness Mean adalah nilai rata-rata estimasi keliling dan luas, kekompakan se digunakan untuk kesalahan standar rata-rata kekompakan, dan nilai rata-rata perhitungan tertinggi diberi nama kekompakan terburuk.')
    st.markdown ('9. Concavity Mean menunjukkan tingkat keparahan bagian cekung dari bentuk, dan concave points mean adalah jumlah bagian cekung dari kontur.')
    st.markdown ('10. Concave Mean adalah jumlah bagian cekung dari kontur.')
    st.markdown ('11. Fractal dimension Mean adalah nilai rata-rata yang dihitung untuk perkiraan garis pantai, kesalahan standar perkiraan garis pantai ditunjukkan sebagai dimensi fraktal se, dan nilai rata-rata tertinggi adalah dimensi fraktal terburuk')
    st.markdown ('12. Concavity Se adalah singkatan dari kesalahan standar bagian cekung, sedangkan titik cekung se adalah singkatan dari kesalahan standar dari bagian cekung bentuk.')