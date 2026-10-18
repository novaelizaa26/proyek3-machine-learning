
import streamlit as st

st.set_page_config(
    page_title="Proyek 3 Machine Learning",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Aplikasi Machine Learning")
st.write("Proyek 3 — Implementasi Model Machine Learning")

st.divider()

pilihan = st.selectbox(
    "Pilih jenis Machine Learning:",
    ["Supervised Learning", "Unsupervised Learning"]
)

if pilihan == "Supervised Learning":
    st.subheader("Prediksi Kepuasan Pelanggan")
    st.write(
        "Aplikasi ini akan digunakan untuk memprediksi "
        "kepuasan pelanggan berdasarkan data yang dimasukkan."
    )
    st.info("Model prediksi akan dihubungkan pada tahap berikutnya.")

else:
    st.subheader("Clustering Berita")
    st.write(
        "Aplikasi ini akan digunakan untuk mengelompokkan "
        "berita berdasarkan kemiripan teks."
    )
    st.info("Model clustering akan dihubungkan pada tahap berikutnya.")

st.divider()
st.caption("Proyek Machine Learning — Kelompok 1")
