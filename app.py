import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from assets import (
    read_data,
    preprocess_and_print,
    pelabelan,
    tf_idf,
    proses_pca,
    fold,
    eval_akhir,
    fold_no,
    preprocess,
    pred_sentimen,
)

if "df" not in st.session_state:
    st.session_state.df = None
if "data" not in st.session_state:
    st.session_state.data = None
if "data_tf" not in st.session_state:
    st.session_state.data_tf = None
if "h_pca" not in st.session_state:
    st.session_state.h_pca = None

st.set_page_config(
    page_title="Kereta Madura",
    page_icon=":chart_with_upwards_trend:",  # Ikoni bisa disesuaikan
    layout="centered",  # Pilihan layout ("wide" atau "centered")
    initial_sidebar_state="expanded",  # Kondisi awal sidebar ("expanded" atau "collapsed")
)

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Home",
            "Dataset",
            "Preprocessing",
            "TF-IDF",
            "PCA",
            "Evaluasi",
            "Sentimen",
        ],
        icons=[
            "house",
            "book",
            "radioactive",
            "bezier2",
            "bounding-box",
            "ubuntu",
            "chat-left-text-fill",
        ],
        menu_icon="menu-up",
        default_index=0,
    )

if selected == "Home":
    st.title("Home")
    st.subheader(
        "Model Naive Bayes dan PCA untuk Data Sentimen Masyarakat (Reaktivasi Kereta Api Madura)"
    )

if selected == "Dataset":
    st.title("Dataset")
    sub_dataset = option_menu(
        menu_title="Pilih Dataset",
        options=["Main Data", "Upload Data"],
        icons=["database", "cloud-arrow-up"],
        default_index=0,
    )

    if sub_dataset == "Main Data":
        st.subheader("Main Data")
        st.session_state.df = pd.read_excel("Dataset.xlsx")
        st.write(st.session_state.df)

    if sub_dataset == "Upload Data":
        st.subheader("Unggah Dataset (Excel)")
        upload_file = st.file_uploader("Pilih file Excel", type=["xlsx"])

        if upload_file is not None:
            st.success("File berhasil diunggah.")
            st.subheader("Preview Data:")
            st.session_state.df = read_data(upload_file)
            if st.session_state.df is not None:
                st.write(st.session_state.df)
        else:
            st.warning("Silakan unggah file excel untuk melihat data.")

if selected == "Preprocessing":
    st.title("Preprocessing")
    sub_preprocessing = option_menu(
        menu_title="Pilih Proeses Preprocessing",
        options=["Main Data Preprocessing", "Manual Preprocessing"],
        icons=["database", "repeat"],
        default_index=0,
    )

    if sub_preprocessing == "Main Data Preprocessing":
        st.subheader("Hasil Preprocessing")
        st.session_state.data = pd.read_excel("Preprocessing.xlsx")
        st.write(st.session_state.data)

    if sub_preprocessing == "Manual Preprocessing":
        if st.button("Proses Data"):
            data = pelabelan(st.session_state.df)
            data["text"] = data["text"].apply(preprocess_and_print)
            st.write(data)
            st.session_state.data = data
            st.success("Proses Data Selesai")

if selected == "TF-IDF":
    st.title("Proses TF-IDF")
    if st.session_state.data is None:
        st.write("Buka Menu Preprocessing Terlebih Dahulu")
    else:
        st.subheader("Data sebelum TF-IDF")
        st.write(st.session_state.data)
        if st.button("Mulai Proses TF-IDF"):
            data = tf_idf(st.session_state.data)
            st.subheader("Hasil TF-IDF")
            st.write(data[0])
            st.session_state.data_tf = data[1]
            st.success("Proses Data Selesai")

if selected == "PCA":
    st.title("Proses PCA")
    if st.session_state.data_tf is None:
        st.write("Lakukan Proses Pada Menu TF-IDF Dahulu!!")
    else:
        n = st.slider("Nilai Component PCA", min_value=1, max_value=10, value=4)
        pca = proses_pca(st.session_state.data_tf, st.session_state.data, n)
        if n < 6:
            col1, col2 = st.columns(2)
            with col1:
                st.header("Hasil PCA")
                st.write(pca[0])

            with col2:
                st.header("Nilai Eigen")
                st.write(pca[1])
        else:
            st.header("Hasil PCA")
            st.write(pca[0])
            st.header("Nilai Eigen")
            st.write(pca[1])
        st.session_state.h_pca = pca[0]

if selected == "Evaluasi":
    st.title("Nilai Evaluasi")
    if st.session_state.h_pca is None:
        st.write("Lakukan Proses PCA pada Menu PCA")
    else:
        n = st.slider("Nilai K", min_value=2, max_value=10, value=4)
        st.subheader("Rata - Rata Evaluasi PCA")
        evaluasi = fold(
            st.session_state.data_tf, st.session_state.data, n, st.session_state.h_pca
        )
        rata_rata = eval_akhir(evaluasi)
        pca1, pca2, pca3, pca4 = st.columns(4)
        with pca1:
            st.write("Akurasi:", round(rata_rata["Akurasi"], 2))
        with pca2:
            st.write("Presisi:", round(rata_rata["Presisi"], 2))
        with pca3:
            st.write("Recall:", round(rata_rata["Recall"], 2))
        with pca4:
            st.write("F1-Score:", round(rata_rata["F1-Score"], 2))

        eval_no = fold_no(st.session_state.data_tf, st.session_state.data, n)
        rata = eval_akhir(eval_no)
        st.subheader("Rata - Rata Evaluasi Tanpa PCA")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            st.write("Akurasi:", round(rata["Akurasi"], 2))
        with pc2:
            st.write("Presisi:", round(rata["Presisi"], 2))
        with pc3:
            st.write("Recall:", round(rata["Recall"], 2))
        with pc4:
            st.write("F1-Score:", round(rata["F1-Score"], 2))

        eval1, eval2 = st.columns(2)
        with eval1:
            st.subheader(f"Nilai K-Fold PCA")
            st.write(evaluasi)
        with eval2:
            st.subheader(f"Nilai K-Fold Tanpa PCA")
            st.write(evaluasi)

if selected == "Sentimen":
    st.title("Sentimen Prediction")
    input_text = st.text_input("Masukkan Kalimat Sentimen:")
    if st.button("Proses"):
        if input_text != "":
            st.subheader("Hasil Prediksi Sentimen")
            st.write(pred_sentimen(input_text))
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Kalimat Input")
                st.write(input_text)
            with col2:
                st.subheader("Hasil Preprocessing")
                st.write(preprocess(input_text))

        else:
            st.subheader("Kalimat Sentimen Tidak Boleh Kosong !!!")
