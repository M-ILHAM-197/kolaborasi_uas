import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Aplikasi Data Mining", page_icon=":computer:", layout="wide")

st.subheader("**M. Ilham Anggis Bangkit Pamungkas**")
st.subheader("**200411100197**")

st.title("Proyek Sains Data :notebook:")

st.write("Klasifikasi Saham")



tab1, tab2, tab3, tab4 = st.tabs(["Data", "Preprocessing Data", "Modelling", "Implementasi"])

with tab1:
    st.header("Data")
    penguin_raw = pd.read_csv('https://raw.githubusercontent.com/M-ILHAM-197/kolaborasi_uas/main/BBRI.JK%20(1).csv')
    st.write(penguin_raw)
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
    penguin_raw = pd.read_csv('https://raw.githubusercontent.com/M-ILHAM-197/kolaborasi_uas/main/BBRI.JK%20(1).csv')
    x = penguin_raw.iloc[:,2:-1]

    st.write("Sebelum dinormalisasi")
    st.write(x.head(10))

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data = min_max_scaler.fit_transform(x)
    dataset = pd.DataFrame(penguin_raw)

    st.write("Setelah dinormalisasi")
    st.write(dataset.head(10))
    
    penguin_raw.loc[(penguin_raw['species'] == "Adelie"), "species"] = 0
    penguin_raw.loc[(penguin_raw['species'] == "Gentoo"), "species"] = 1
    penguin_raw.loc[(penguin_raw['species'] == "Chinstrap"), "species"] = 2

    penguin_raw.loc[(penguin_raw['island'] == "Torgersen"), "island"] = 0
    penguin_raw.loc[(penguin_raw['island'] == "Biscoe"), "island"] = 1
    penguin_raw.loc[(penguin_raw['island'] == "Dream"), "island"] = 2

    penguin_raw['sex'] = penguin_raw['sex'].replace({'male':1, 'female':0})

    data = penguin_raw.drop(columns=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"])

    st.write("Setelah di encoding")
    st.dataframe(data.head(10))

    st.subheader("Preprocessing Data Berhasil")

    dataset1 = data
    dataset1 = dataset1.join(dataset)
    
with tab3:
    st.header("Modelling")
    
    randomforest_cekbox = st.checkbox("Random Forest")
    knn_cekbox = st.checkbox("KNN")
    bayes_gaussian_cekbox = st.checkbox("Naive-Bayes Gaussian")
    decission3_cekbox = st.checkbox("Decission Tree")

    #=========================== Spliting data ======================================
    X = dataset.iloc[:,2:-1]
    Y = penguin_raw.iloc[:,-1]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)    

    #============================ Model =================================

    randomforest = RandomForestClassifier()
    randomforest.fit(X_train, Y_train)
    y_pred_randomforest = randomforest.predict(X_test)
    randomforest_accuracy = round(100 * accuracy_score(Y_test, y_pred_randomforest), 2)
    #===================== KNN =======================

    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    y_predknn = knn.predict(X_test)
    knn_accuracy = round(100 * accuracy_score(Y_test, y_predknn), 2)

    #===================== Bayes Gaussian =============
    gaussian = GaussianNB()
    gaussian.fit(X_train,Y_train)
    y_pred_gaussian   =  gaussian.predict(X_test)
    gauss_accuracy  = round(100*accuracy_score(Y_test, y_pred_gaussian),2)
    gaussian_eval = classification_report(Y_test, y_pred_gaussian,output_dict = True)
    gaussian_eval_df = pd.DataFrame(gaussian_eval).transpose()

    #===================== Decission tree =============
    decission3  = DecisionTreeClassifier(criterion="gini")
    decission3.fit(X_train,Y_train)
    y_pred_decission3 = decission3.predict(X_test)
    decission3_accuracy = round(100*accuracy_score(Y_test, y_pred_decission3),2)
    decission3_eval = classification_report(Y_test, y_pred_decission3,output_dict = True)
    decission3_eval_df = pd.DataFrame(decission3_eval).transpose()

    st.markdown("---")

    #===================== Cek Box ====================
    if randomforest_cekbox:
        st.write("##### Random Forest Classifier")
        st.error("Dengan menggunakan metode Random Forest didapatkan akurasi sebesar:")
        st.error(f"Akurasi = {randomforest_accuracy}%")
        st.markdown("---")

    if knn_cekbox:
        st.write("##### KNN")
        st.warning("Dengan menggunakan metode KNN didapatkan akurasi sebesar:")
        # st.warning(knn_accuracy)
        st.warning(f"Akurasi  =  {knn_accuracy}%")
        st.markdown("---")

    if bayes_gaussian_cekbox:
        st.write("##### Naive Bayes Gausssian")
        st.info("Dengan menggunakan metode Bayes Gaussian didapatkan hasil akurasi sebesar:")
        st.info(f"Akurasi = {gauss_accuracy}%")
        st.markdown("---")

    if decission3_cekbox:
        st.write("##### Decission Tree")
        st.success("Dengan menggunakan metode Decission tree didapatkan hasil akurasi sebesar:")
        st.success(f"Akurasi = {decission3_accuracy}%")

with tab4:
    st.header("Implementasi")

    def user_input_features():
        island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.selectbox('Sex',('male','female'))
        bill_length_mm = st.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        
        features = pd.DataFrame(data, index=[0])
        return features
        
    input_df = user_input_features()

    penguins_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    penguins = penguins_raw.drop(columns=['species'], axis=1)
    df = pd.concat([input_df,penguins],axis=0)

    encode = ['sex','island']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]
    df = df[:1]

    st.subheader("Hasil :")
    st.write(df)

    load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)

    st.subheader('Prediksi')
    penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
    st.write(penguins_species[prediction])

    st.subheader('Probabilitas Prediksi')
    st.write(prediction_proba)