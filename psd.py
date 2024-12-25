import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

def main():
    with st.sidebar:
        page = option_menu("Pilih Halaman", [
                           "Home", "Data Understanding", "Preprocessing", "Model", "Evaluasi", "Testing"], default_index=0)

    if page == "Home":
        show_home()
    elif page == "Data Understanding":
        show_understanding()
    elif page == "Preprocessing":
        show_preprocessing()
    elif page == "Model":
        show_model()
    elif page == "Evaluasi":
        show_evaluasi()
    elif page == "Testing":
        show_testing()

def show_home():
    st.title("Klasifikasi Prediksi Kelulusan Mahasiswa dengan Metode K-Nearest Neighbors")
    st.header("Apa itu K-Nearest Neighbor?")
    st.write("K-Nearest Neighbor (KNN) merupakan algoritma yang digunakan untuk memprediksi kelas atau kategori dari data baru berdasarkan mayoritas kelas dari tetangga terdekat")
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses pengolahan data dan klasifikasi dengan menggunakan metode KNN.")
    st.header("Data")
    st.write("Data yang digunakan adalah Dataset Prediksi Kelulusan Mahasiswa.")

def show_understanding():
    st.title("Data Understanding")
    data = pd.read_excel("Graduation_Prediction.xlsx", sheet_name="Worksheet")
    st.header("Metadata dari Dataset")
    st.dataframe(data)

    col1, col2 = st.columns(2, vertical_alignment='top')
    with col1:
        st.write("Jumlah Data : ", len(data.axes[0]))
        st.write("Jumlah Atribut : ", len(data.axes[1]))

    with col2:
        st.write(f"Terdapat {len(data['Target'].unique())} Label Kelas, yaitu : {data['Target'].unique()}")

    st.markdown("---")
    st.header("Tipe Data & Missing Value")

    r2col1, r2col2 = st.columns(2, vertical_alignment='bottom')
    with r2col1:
        st.write("Tipe Data")
        st.write(data.dtypes)

    with r2col2:
        st.write("Missing Value")
        st.write(data.isnull().sum())

    st.markdown("---")
    st.header("Distribusi Target")
    target_counts = data['Target'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts.plot(kind='bar', color='blue', ax=ax)
    ax.set_title('Distribusi Target')
    ax.set_xlabel('Target')
    ax.set_ylabel('Jumlah')
    st.pyplot(fig)

def show_preprocessing():
    st.title("Preprocessing")
    data = pd.read_excel("Graduation_Prediction.xlsx", sheet_name="Worksheet")
    st.header("Memilih Atribut yang digunakan untuk Pemodelan")
    st.dataframe(data)
    st.markdown("---")
    st.header("Label Encoding untuk Fitur Kategorikal")

    # Melakukan encoding pada fitur kategorikal menggunakan LabelEncoder
    label_encoder = LabelEncoder()
    for col in ['Marital status', 'Gender', 'Target']:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])
            st.write(f"Fitur {col} setelah Label Encoding :")
            st.write(data[col].value_counts())

    st.markdown("---")
    # Normalisasi Data menggunakan Min-Max Scaler
    st.header("Normalisasi Data menggunakan Min Max Scalar")

    # Memisahkan fitur dan target
    x = data.drop(['Target'], axis=1)
    y = data['Target']

    # Normalisasi data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

    # Menampilkan data setelah normalisasi
    st.dataframe(x_scaled)

    # Menyimpan data yang telah dinormalisasi dalam session state
    st.session_state['preprocessed_data'] = x_scaled
    st.session_state['Target'] = y

def show_model():
    st.title("Pemodelan")

    if 'preprocessed_data' in st.session_state and 'Target' in st.session_state:
        X_scaled = st.session_state['preprocessed_data']
        y = st.session_state['Target']
        combined_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        st.dataframe(combined_data)

        st.markdown("---")
        st.header("Memilih Fitur untuk Pemodelan")
        
        # Menambahkan fitur seleksi
        selected_features = st.multiselect("Pilih fitur yang ingin digunakan:", options=X_scaled.columns.tolist(), default=X_scaled.columns.tolist())

        # Memisahkan data berdasarkan fitur yang dipilih
        X_selected = X_scaled[selected_features]

        st.markdown("---")
        st.header("Memecah menjadi data Training dan data Testing")

        # Memisahkan data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, random_state=0, train_size=0.8, shuffle=True)

        # Menyimpan data untuk evaluasi
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

        # Menampilkan informasi tentang data training dan testing
        st.write("Data Training:")
        st.dataframe(X_train)
        st.write("Data Testing:")
        st.dataframe(X_test)

def show_evaluasi():
    st.title("Evaluasi Model")

    if 'X_train' in st.session_state:
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']

        models = {
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "SVM": SVC(kernel='linear', probability=True),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }

            st.subheader(f"{name} - Metrics")
            st.write(f"Accuracy: {acc:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            st.pyplot(fig)

        # Tampilkan tabel evaluasi tanpa kolom Confusion Matrix
        results_df = pd.DataFrame(results).T
        st.subheader("Tabel Evaluasi Model")
        st.dataframe(results_df)

        # Visualisasi hubungan antar metrik
        st.subheader("Hubungan Antar Metrik")
        metrics_data = results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']]
        fig = sns.pairplot(metrics_data)
        st.pyplot(fig)

        best_model = max(results.items(), key=lambda x: x[1]['Accuracy'])
        st.subheader("Model Terbaik")
        st.write(f"Model terbaik adalah {best_model[0]} dengan akurasi {best_model[1]['Accuracy']:.2f}")

        incorrect = X_test[y_test != y_pred]
        st.write("Contoh kesalahan prediksi:")
        st.dataframe(incorrect)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        st.write("Cross-validation scores:", cv_scores)
        st.write("Mean CV score:", np.mean(cv_scores))

def show_testing():
    st.title("Testing Model")

    if 'X_train' in st.session_state:
        st.header("Input Data untuk Prediksi")
        input_data = []

        for col in ['Marital status', 'Course', 'Daytime/evening attendance', 'Previous qualification',
                    'Educational special needs', 'Debtor', 'Gender', 'Scholarship holder', 'Age at enrollment',
                    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
                    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
                    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
                    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
                    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
                    'Unemployment rate', 'Inflation rate', 'GDP']:
            value = st.number_input(f"{col}", step=1.0)
            input_data.append(value)

        if st.button('Predict'):
            input_data = np.array(input_data).reshape(1, -1)
            model = RandomForestClassifier(random_state=42)
            model.fit(st.session_state['X_train'], st.session_state['y_train'])
            result = model.predict(input_data)
            st.write("Hasil Prediksi: ", "Lulus" if result[0] == 1 else "Dropout")

if __name__ == "__main__":
    st.set_page_config(page_title="Klasifikasi Kelulusan", page_icon="img/knn.png")
    main()
