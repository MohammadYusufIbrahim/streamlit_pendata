import streamlit as st

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


import pickle

from sklearn import metrics

st.set_page_config(
    page_title="Prediksi Penyakit Jantung"
)

st.title('Prediksi Penyakit jantung')
st.write("""
Aplikasi Untuk Memprediksi Kemungkinan Penyakit Jantung
""")
st.write("""
Nama : Muhammad Yusuf Ibrahim
""")
st.write("""
NIM : 210411100095
""")

tab1, tab2, tab3, tab4 = st.tabs(["Data Understanding", "Preprocessing", "Modelling", "Implementation"])

with tab1:
    st.write("""
    <h5>Data Understanding</h5>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Dataset:
    <a href="https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci"> https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Repository Github
    https://raw.githubusercontent.com/MohammadYusufIbrahim/databack/main/heart.csv
    """, unsafe_allow_html=True)
    
    st.write('Type dataset ini adalah campuran (Numerik dan Boolean)')
    st.write('Dataset ini berisi tentang klasifikasi penyakit jantung')
    df = pd.read_csv("https://raw.githubusercontent.com/MohammadYusufIbrahim/databack/main/heart.csv")
    st.write("Dataset Heart Disease : ")
    st.write(df)
    st.write("Penjelasan kolom-kolom yang ada")

    st.write("""
    <ol>
    <li>age : Umur dalam satuan Tahun</li>
    <li>sex : Jenis Kelamin (1=Laki-laki, 0=Perempuan)</li>
    <li>cp : chest pain type (tipe sakit dada)(0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)</li>
    <li>trestbps : tekanan darah saat dalam kondisi istirahat dalam mm/Hg</li>
    <li>chol : serum sholestoral (kolestrol dalam darah) dalam Mg/dl </li>
    <li>fbs : fasting blood sugar (kadar gula dalam darah setelah berpuasa) lebih dari 120 mg/dl (1=Iya, 0=Tidak)</li>
    <li>restecg : hasil test electrocardiographic (0 = normal, 1 = memiliki kelainan gelombang ST-T (gelombang T inversi dan/atau ST elevasi atau depresi > 0,05 mV), 2 = menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes)</li>
    <li>thalach : rata-rata detak jantung pasien dalam satu menit</li>
    <li>exang :  keadaan dimana pasien akan mengalami nyeri dada apabila berolah raga, 0 jika tidak nyeri, dan 1 jika menyebabkan nyeri</li>
    <li>oldpeak : depresi ST yang diakibatkan oleh latihan relative terhadap saat istirahat</li>
    <li>Slope : slope dari puncak ST setelah berolah raga. Atribut ini memiliki 3 nilai yaitu 0 untuk downsloping, 1 untuk flat, dan 2 untuk upsloping.</li>
    <li>Ca: banyaknya pembuluh darah yang terdeteksi melalui proses pewarnaan flourosopy</li>
    <li>Thal: detak jantung pasien. Atribut ini memiliki 3 nilai yaitu 1 untuk fixed defect, 2 untuk normal dan 3 untuk reversal defect</li>
    <li>target: hasil diagnosa penyakit jantung, 0 untuk terdiagnosa positif terkena penyakit jantung koroner, dan 1 untuk negatif terkena penyakit jantung koroner.</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("""
    <h5>Preprocessing Data</h5>
    <br>
    """, unsafe_allow_html=True)
    st.write("""
    <p style="text-align: justify;text-indent: 45px;">Preprocessing data adalah proses mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini diperlukan untuk memperbaiki kesalahan pada data mentah yang seringkali tidak lengkap dan memiliki format yang tidak teratur. Preprocessing melibatkan proses validasi dan imputasi data.</p>
    <p style="text-align: justify;text-indent: 45px;">Salah satu tahap Preprocessing data adalah Normalisasi. Normalisasi data adalah elemen dasar data mining untuk memastikan record pada dataset tetap konsisten. Dalam proses normalisasi diperlukan transformasi data atau mengubah data asli menjadi format yang memungkinkan pemrosesan data yang efisien.</p>
    <br>
    """,unsafe_allow_html=True)
    scaler = st.radio(
    "Pilih metode normalisasi data",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df, columns = ['age','trestbps','chol','thalach','oldpeak'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['age','trestbps','chol','thalach','oldpeak'])
        df_drop_column_for_minmaxscaler=df.drop(['age','trestbps','chol','thalach','oldpeak'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    st.write(df_new)

with tab3:
    st.write("""
    <h5>Modelling</h5>
    <br>
    """, unsafe_allow_html=True)

    nb = st.checkbox("Naive Bayes")  # Checkbox for Naive Bayes
    knn = st.checkbox("KNN")  # Checkbox for KNN
    ds = st.checkbox("Decision Tree")  # Checkbox for Decision Tree
    mlp = st.checkbox("MLP")  # Checkbox for MLP

    # Splitting the data into features and target variable
    X = df.drop('target', axis=1)
    y = df['target']

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = []  # List to store selected models

    if nb:
        models.append(('Naive Bayes', GaussianNB()))
    if knn:
        models.append(('KNN', KNeighborsClassifier()))
    if ds:
        models.append(('Decision Tree', DecisionTreeClassifier()))
    if mlp:
        models.append(('MLP', MLPClassifier()))

    if len(models) == 0:
        st.warning("Please select at least one model.")

    else:
        accuracy_scores = []  # List to store accuracy scores

        st.write("<h6>Accuracy Scores:</h6>", unsafe_allow_html=True)
        st.write("<table><tr><th>Model</th><th>Accuracy</th></tr>", unsafe_allow_html=True)

        for model_name, model in models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            st.write("<tr><td>{}</td><td>{:.2f}</td></tr>".format(model_name, accuracy), unsafe_allow_html=True)

        st.write("</table>", unsafe_allow_html=True)

        # Displaying the table of test labels and predicted labels
        st.write("<h6>Test Labels and Predicted Labels:</h6>", unsafe_allow_html=True)
        labels_df = pd.DataFrame({'Test Labels': y_test, 'Predicted Labels': y_pred})
        st.write(labels_df)


# Define the decision tree classifier model
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Save the decision tree model as a pickle file
filename = 'decision_tree.pkl'
pickle.dump(model, open(filename, 'wb'))

with tab4:
    st.write("""
    <h5>Model Terbaik yaitu Decision Tree dengan akurasi 0,99</h5>
    <br>
    """, unsafe_allow_html=True)
    st.write("""
    <h5>Implementation</h5>
    <br>
    """, unsafe_allow_html=True)
    X=df_new.iloc[:,0:13].values
    y=df_new.iloc[:,13].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)

    age=st.number_input("umur : ")
    sex=st.selectbox(
        'Pilih Jenis Kelamin',
        ('Laki-laki','Perempuan')
    )
    if sex=='Laki-laki':
        sex=1
    elif sex=='Perempuan':
        sex=0
    cp=st.selectbox(
        'Jenis nyeri dada',
        ('Typical Angina','Atypical angina','non-anginal pain','asymptomatic')
    )
    if cp=='Typical Angina':
        cp=0
    elif cp=='Atypical angina':
        cp=1
    elif cp=='non-anginal pain':
        cp=2
    elif cp=='asymptomatic':
        cp=3
    trestbps=st.number_input('resting blood pressure / tekanan darah saat kondisi istirahat(mm/Hg)')
    chol=st.number_input('serum cholestoral / kolestrol dalam darah (Mg/dl)')
    fbs=st.selectbox(
        'fasting blood sugar / gula darah puasa',
        ('Dibawah 120', 'Diatas 120')
    )
    if fbs=='Dibawah 120':
        fbs=0
    elif fbs=='Diatas 120':
        fbs=1
    restecg=st.selectbox(
        'resting electrocardiographic results',
        ('normal','mengalami kelainan gelombang ST-T','menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes')    
    )
    if restecg=='normal':
        restecg=0
    elif restecg=='mengalami kelainan gelombang ST-T':
        restecg=1
    elif restecg=='menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes':
        restecg=2
    thalach=st.number_input('thalach (rata-rata detak jantung pasien dalam satu menit)')
    exang=st.selectbox(
        'exang/exercise induced angina',
        ('ya','tidak')
    )
    if exang=='ya':
        exang=1
    elif exang=='tidak':
        exang=0
    oldpeak=st.number_input('oldpeak/depresi ST yang diakibatkan oleh latihan relative terhadap saat istirahat')
    slope=st.selectbox(
        'slope of the peak exercise',
        ('upsloping','flat','downsloping')
    )
    if slope=='upsloping':
        slope=0
    elif slope=='flat':
        slope=1
    elif slope=='downsloping':
        slope=2
    ca=st.number_input('number of major vessels')
    thal=st.selectbox(
        'Thalassemia',
        ('normal','cacat tetap','cacat reversibel')
    )
    if thal=='normal':
        thal=0
    elif thal=='cacat tetap':
        thal=1
    elif thal=='cacat reversibel':
        thal=2
    
    algoritma2 = st.selectbox(
        'Model Terbaik: ',
        ('Decision Tree','Decision Tree')
    )
    model2 = DecisionTreeClassifier()
    filename2 = 'decision_tree.pkl'

    algoritma = st.selectbox(
        'pilih model klasifikasi lain :',
        ('KNN','Naive Bayes', 'MLP')
    )
    prediksi=st.button("Diagnosis")
    if prediksi:
        if algoritma=='KNN':
            model = KNeighborsClassifier(n_neighbors=3)
            filename='knn.pkl'
        elif algoritma=='Naive Bayes':
            model = GaussianNB()
            filename='gaussian.pkl'
        elif algoritma=='MLP':
            model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
            filename='mlp.pkl'
        
        model2.fit(X_train, y_train)
        Y_pred2 = model2.predict(X_test) 

        score2=metrics.accuracy_score(y_test,Y_pred2)

        loaded_model2 = pickle.load(open(filename2, 'rb'))

        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test) 

        score=metrics.accuracy_score(y_test,Y_pred)

        loaded_model = pickle.load(open(filename, 'rb'))
        if scaler == 'Tanpa Scaler':
            dataArray = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            age_proceced = (age - df['age'].mean()) / df['age'].std()
            trestbps_proceced = (trestbps - df['trestbps'].mean()) / df['trestbps'].std()
            chol_proceced = (chol - df['chol'].mean()) / df['chol'].std()
            thalach_proceced = (thalach - df['thalach'].mean()) / df['thalach'].std()
            oldpeak_proceced = (oldpeak - df['oldpeak'].mean()) / df['oldpeak'].std()
            dataArray = [
                age_proceced, trestbps_proceced, chol_proceced, thalach_proceced, oldpeak_proceced,
                sex, cp, fbs, restecg, exang, slope, ca, thal
            ]

        pred = loaded_model.predict([dataArray])
        pred2 = loaded_model2.predict([dataArray])

        st.write('--------')
        st.write('Hasil dengan Decision Tree :')
        if int(pred2[0])==1:
            st.success(f"Hasil Prediksi : Tidak memiliki penyakit Jantung")
        elif int(pred2[0])==0:
            st.error(f"Hasil Prediksi : Memiliki penyakit Jantung")

        st.write(f"akurasi : {score2}")
        st.write('--------')
        st.write('Hasil dengan ',{algoritma},' :')
        if int(pred[0])==0:
            st.success(f"Hasil Prediksi : Tidak memiliki penyakit Jantung")
        elif int(pred[0])==1:
            st.error(f"Hasil Prediksi : Memiliki penyakit Jantung")

        st.write(f"akurasi : {score}")
