import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from datetime import datetime

st.set_page_config(page_title="Aplikasi Data Mining", page_icon=":computer:", layout="wide")

st.subheader("**M. Ilham Anggis Bangkit Pamungkas**")
st.subheader("**200411100197**")
st.subheader("**Baihaki**")
st.subheader("**200411100181**")

st.title("Proyek Sains Data :notebook:")

st.write("Klasifikasi Saham")



tab1, tab2, tab3, tab4 = st.tabs(["Data", "Preprocessing Data", "Modelling", "Implementasi"])

with tab1:
    st.header("Data")
    data = pd.read_csv('https://raw.githubusercontent.com/M-ILHAM-197/kolaborasi_uas/main/BBRI.JK%20(1).csv')
    st.write(data)
    st.subheader("Penjelasan :")
    st.write("""
            Jelaskan datanya dari mana
    """)
    st.write("""
            Type datanya dari mana
    """)

with tab2:
    st.header("Preprocessing Data")
    st.write("""
         Preprocessing adalah teknik penambangan data yang digunakan untuk mengubah data mentah dalam format yang berguna dan efisien.
    """)
    data = pd.read_csv('https://raw.githubusercontent.com/M-ILHAM-197/kolaborasi_uas/main/BBRI.JK%20(1).csv')
    
    features = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    target = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    st.write("Sebelum dinormalisasi")
    st.write(data.head(10))

    st.write("Min Max Scaler")
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    st.write(features_scaled)

    st.write("Reduksi Dimensi PCA")
    pca = PCA(n_components=4)
    features_pca = pca.fit_transform(features_scaled)
    st.write(features_pca)


    st.subheader("Preprocessing Data Berhasil")


    
with tab3:
    st.header("Modelling")
    
    knn_cekbox = st.checkbox("KNN")
    decission3_cekbox = st.checkbox("Decission Tree")

    #=========================== Spliting data ======================================


    #============================ Model =================================

    #===================== KNN =======================

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Membuat model KNN
    knn_model = KNeighborsRegressor(n_neighbors=5)

    # Melatih model KNN
    knn_model.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = knn_model.predict(X_test)

    # Menghitung MAPE
    mape_knn = mean_absolute_percentage_error(y_test, y_pred)

    #===================== Bayes Gaussian =============


    #===================== Decission tree =============
    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)

    # Membuat model Decision Tree Regression
    dt_model = DecisionTreeRegressor(random_state=42)

    # Melatih model Decision Tree Regression
    dt_model.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = dt_model.predict(X_test)

    # Menghitung MAPE
    mape_decision = mean_absolute_percentage_error(y_test, y_pred)

    st.markdown("---")

    #===================== Cek Box ====================

    if knn_cekbox:
        st.write("##### KNN")
        st.warning("Prediksi menggunakan KNN:")
        # st.warning(knn_accuracy)
        st.warning(f"MAPE  =  {mape_knn}")
        st.markdown("---")


    if decission3_cekbox:
        st.write("##### Decission Tree")
        st.success("Dengan menggunakan metode Decission tree didapatkan hasil akurasi sebesar:")
        st.success(f"Akurasi = {mape_decision}")

with tab4:
    st.header("Implementasi")


    # Menambahkan konten prediksi
    st.header("Prediksi Saham")
    st.write("""
        Masukkan tanggal untuk melakukan prediksi harga saham.
    """)

    # Membaca input tanggal
    date_input = st.date_input("Tanggal", datetime.now())

    # Mengubah input tanggal menjadi fitur
    date_features = scaler.transform(array([[0, 0, 0, date_input.day, date_input.month, date_input.year]]))

    # Melakukan reduksi dimensi pada fitur tanggal
    date_features_pca = pca.transform(date_features)

    # Melakukan prediksi menggunakan model KNN
    knn_prediction = knn_model.predict(date_features_pca)[0]

    # Melakukan prediksi menggunakan model Decision Tree
    dt_prediction = dt_model.predict(date_features_pca)[0]

    st.subheader("Hasil Prediksi Saham")
    st.write("Prediksi menggunakan model KNN:")
    st.write(f"Harga saham: {knn_prediction}")

    st.write("Prediksi menggunakan model Decision Tree:")
    st.write(f"Harga saham: {dt_prediction}")
