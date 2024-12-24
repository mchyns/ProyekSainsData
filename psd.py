import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go

def load_data(file):
    """
    Load and validate data from uploaded file   
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('Graduation_Prediction.xls', 'Graduation_Prediction.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error("Format file tidak didukung. Gunakan CSV atau Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error saat membaca file: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the data:
    - Handle missing values
    - Convert categorical to numerical
    - Scale numerical features
    """
    # Copy dataframe
    df_clean = df.copy()
    

    
    # Handle missing values
    df_clean = df_clean.dropna()
    
    # List of expected columns
    expected_categorical = [
        'Marital status', 'Course', 'Daytime/evening attendance',
        'Previous qualification', 'Educational special needs', 'Debtor', 
        'Gender', 'Scholarship holder'
    ]
    
    expected_numerical = [
        'Age at enrollment', 
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ]
    
    # Cek kolom yang tidak ada
    missing_columns = [col for col in expected_categorical + expected_numerical 
                      if col not in df_clean.columns]
    if missing_columns:
        st.warning(f"Kolom berikut tidak ditemukan dalam dataset: {', '.join(missing_columns)}")
    
    # Filter kolom yang ada
    categorical_columns = [col for col in expected_categorical if col in df_clean.columns]
    numerical_columns = [col for col in expected_numerical if col in df_clean.columns]
    
    if not categorical_columns and not numerical_columns:
        st.error("Tidak ada kolom yang sesuai dalam dataset. Mohon periksa nama kolom pada dataset Anda.")
        return None, None, None, None, None
    
    # Label Encoding untuk variabel kategorikal
    label_encoders = {}
    for column in categorical_columns:
        try:
            label_encoders[column] = LabelEncoder()
            df_clean[column] = label_encoders[column].fit_transform(df_clean[column])
        except Exception as e:
            st.error(f"Error saat mengenkode kolom {column}: {str(e)}")
            return None, None, None, None, None
    
    # Scale fitur numerik
    scaler = StandardScaler()
    if numerical_columns:
        try:
            df_clean[numerical_columns] = scaler.fit_transform(df_clean[numerical_columns])
        except Exception as e:
            st.error(f"Error saat scaling kolom numerik: {str(e)}")
            return None, None, None, None, None
    
    return df_clean, scaler, label_encoders, categorical_columns, numerical_columns

def train_models(X, y, scaler, k=5):
    """
    Train multiple models and return their metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=k),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        # Cek dan skala X_test
        if X_test.isnull().values.any():
            st.error("X_test mengandung nilai yang hilang. Silakan periksa data Anda.")
            return

        # Pastikan kolom X_test sesuai dengan X_train
        X_test = X_test[X_train.columns.intersection(X_test.columns)]  # Mengatur ulang kolom X_test

        # Terapkan scaler
        try:
            X_test_scaled = scaler.transform(X_test)  # Terapkan scaler
        except ValueError as e:
            st.error(f"Error saat menerapkan scaler: {str(e)}")
            return

        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report,
            'conf_matrix': conf_matrix,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

    return results

def plot_confusion_matrix(conf_matrix, model_name):
    """
    Plot confusion matrix menggunakan plotly
    """
    labels = ['Dropout/Enrolled', 'Graduate']
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=labels,
        y=labels,
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=500
    )
    
    return fig

def plot_feature_importance(model, X):
    """
    Plot feature importance menggunakan permutation importance
    """
    try:
        # Dapatkan prediksi numerik dari model
        y_pred = model.predict(X)
        
        # Hitung permutation importance
        r = permutation_importance(
            model, X, y_pred,
            n_repeats=10,
            random_state=42,
            scoring='accuracy'
        )
        
        # Buat DataFrame untuk importance scores
        feature_importance = pd.DataFrame(
            {'feature': X.columns,
             'importance': r.importances_mean}
        )
        
        # Urutkan berdasarkan importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Buat visualisasi dengan plotly
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance (Permutation)',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        
        # Update layout
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error dalam menghitung feature importance: {str(e)}")
        st.write("Detail error:", str(e))
        return None

def plot_metrics_comparison(results):
    """
    Plot comparison of multiple metrics between models
    """
    # Prepare data for plotting
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.extend([
            {'Model': model_name, 'Metric': 'Akurasi', 'Value': result['accuracy']},
            {'Model': model_name, 'Metric': 'Presisi', 'Value': result['precision']},
            {'Model': model_name, 'Metric': 'Recall', 'Value': result['recall']},
            {'Model': model_name, 'Metric': 'F1-Score', 'Value': result['f1']}
        ])
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create grouped bar plot
    fig = px.bar(
        df_metrics,
        x='Metric',
        y='Value',
        color='Model',
        title='Grafik Hasil Evaluasi',
        barmode='group',
        labels={'Value': 'Skor', 'Metric': 'Metrik'}
    )
    
    return fig

def plot_correlation_matrix(X):
    """
    Plot correlation matrix untuk semua fitur
    """
    # Hitung correlation matrix
    corr_matrix = X.corr()
    
    # Buat heatmap dengan plotly
    fig = px.imshow(
        corr_matrix,
        title='Correlation Matrix (Heatmap) Antar Fitur',
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu_r',  # Red-Blue color scale
        aspect='auto'
    )
    
    # Update layout
    fig.update_layout(
        width=800,
        height=800,
        xaxis_title="Fitur",
        yaxis_title="Fitur"
    )
    
    # Tambahkan anotasi nilai korelasi
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=i,
                y=j,
                text=str(round(corr_matrix.iloc[i, j], 2)),
                showarrow=False,
                font=dict(color="black")
            )
    
    return fig

def plot_metrics_relationships(results):
    """
    Plot hubungan antar metrics untuk semua model
    """
    # Prepare data
    metrics_data = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1']
        }
        for name, res in results.items()
    ])
    
    # Create scatter matrix plot
    fig = px.scatter_matrix(
        metrics_data,
        dimensions=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        color='Model',
        title='Hubungan Antar Metrics'
    )
    
    # Update layout
    fig.update_layout(
        width=800,
        height=800,
    )
    
    return fig

def plot_feature_target_relationship(X, y):
    """
    Plot hubungan antara fitur numerik dengan target
    """
    # Convert target to numeric if needed
    y_numeric = pd.to_numeric(y, errors='coerce')
    
    # Select numerical columns only
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Calculate correlations
    correlations = []
    for column in X_numeric.columns:
        corr = np.corrcoef(X_numeric[column], y_numeric)[0,1]
        correlations.append({'Feature': column, 'Correlation': corr})
    
    # Create DataFrame and sort
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    
    # Create bar plot
    fig = px.bar(
        corr_df,
        x='Feature',
        y='Correlation',
        title='Korelasi Fitur dengan Target',
        labels={'Correlation': 'Korelasi dengan Target'},
    )
    
    # Update layout
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig

def plot_prediction_distribution(results):
    """
    Plot pie chart distribusi prediksi
    """
    # Ambil prediksi dari model terbaik
    best_model = max(results.items(), key=lambda x: x[1]['f1'])[1]
    predictions = best_model['y_pred']
    
    # Hitung jumlah masing-masing kelas
    unique, counts = np.unique(predictions, return_counts=True)
    labels = ['Dropout/Enrolled' if x == 0 else 'Graduate' for x in unique]
    
    # Buat pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts,
        hole=.3,
        title='Distribusi Prediksi'
    )])
    
    return fig

def plot_feature_scatter(X, y, feature1, feature2):
    """
    Plot scatter plot untuk dua fitur
    """
    fig = px.scatter(
        x=X[feature1],
        y=X[feature2],
        color=y,
        title=f'Hubungan antara {feature1} dan {feature2}',
        labels={
            'x': feature1,
            'y': feature2,
            'color': 'Status'
        },
        color_discrete_map={
            0: 'red',
            1: 'green'
        }
    )
    
    return fig

def plot_accuracy_comparison(results):
    """
    Plot bar chart perbandingan akurasi
    """
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=accuracies,
            text=[f'{acc:.2%}' for acc in accuracies],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Perbandingan Akurasi Model',
        xaxis_title='Model',
        yaxis_title='Akurasi',
        yaxis_tickformat=',.0%'
    )
    
    return fig

def main():
    st.title('Sistem Prediksi Performa Akademik Mahasiswa')
    st.write('Aplikasi ini membandingkan berbagai metode machine learning untuk memprediksi performa akademik mahasiswa')
    
    # Sidebar menu
    menu = st.sidebar.selectbox("Pilih Menu", ["Tampilkan dan Proses Data", "Training Model", "Prediksi Performa"])
    
    # Initialize session state
    if 'training_done' not in st.session_state:
        st.session_state['training_done'] = False
    if 'results_displayed' not in st.session_state:
        st.session_state['results_displayed'] = False
    
    if menu == "Training Model":
        st.header('2. Training Model')
        if 'df_clean' in st.session_state:
            df_clean = st.session_state['df_clean']
            features = st.session_state['categorical_columns'] + st.session_state['numerical_columns']
            X = df_clean[features]
            y = df_clean['Target']
            k = st.slider('Pilih nilai K untuk KNN:', min_value=1, max_value=20, value=5)
            
            if st.button('Train Models') or st.session_state['training_done']:
                if not st.session_state['training_done']:
                    results = train_models(X, y, st.session_state['scaler'], k)  # Pass scaler here
                    if results:  # Pastikan results tidak kosong
                        st.session_state['results'] = results
                        st.session_state['training_done'] = True
                    else:
                        st.error("Tidak ada hasil yang dihasilkan dari pelatihan model.")
        
        # Tampilkan data dalam bentuk tabel
        st.write("Data Preview:")
        st.dataframe(df)

        # Tampilkan kolom yang tersedia dalam bentuk tabel
        columns_df = pd.DataFrame(df.columns.tolist(), columns=["Kolom"])
        st.write("Kolom yang tersedia dalam dataset:")
        st.dataframe(columns_df.T)  # Tampilkan tabel secara horizontal

        # Proses data
        df_clean, scaler, label_encoders, categorical_columns, numerical_columns = preprocess_data(df)
        if df_clean is not None:
            st.write("Statistik Data:")
            st.dataframe(df_clean.describe())
            st.session_state['df_clean'] = df_clean  # Simpan dataframe bersih ke session state
            st.session_state['scaler'] = scaler
            st.session_state['label_encoders'] = label_encoders
            st.session_state['categorical_columns'] = categorical_columns
            st.session_state['numerical_columns'] = numerical_columns
            
            # Plot correlation matrix (heatmap)
            st.subheader("Heatmap Korelasi Antar Fitur:")
            corr_matrix_plot = plot_correlation_matrix(df_clean[categorical_columns + numerical_columns])
            st.plotly_chart(corr_matrix_plot)

            # Plot feature-target relationship
            st.subheader("Hubungan Fitur dengan Target:")
            feature_target_plot = plot_feature_target_relationship(df_clean[categorical_columns + numerical_columns], df_clean['Target'])
            st.plotly_chart(feature_target_plot)

 if menu == "Training Model":
    st.header('2. Training Model')
    if 'df_clean' in st.session_state:
        df_clean = st.session_state['df_clean']
        features = st.session_state['categorical_columns'] + st.session_state['numerical_columns']
        X = df_clean[features]
        y = df_clean['Target']
        k = st.slider('Pilih nilai K untuk KNN:', min_value=1, max_value=20, value=5)
        
        if st.button('Train Models') or st.session_state['training_done']:
            if not st.session_state['training_done']:
                results = train_models(X, y, st.session_state['scaler'], k)  # Pass scaler here
                if results:  # Pastikan results tidak kosong
                    st.session_state['results'] = results
                    st.session_state['training_done'] = True
                else:
                    st.error("Tidak ada hasil yang dihasilkan dari pelatihan model.")
                
                results = st.session_state['results']
                
                # Tambahkan visualisasi confusion matrix untuk setiap model
                st.subheader('Confusion Matrix:')
                for model_name, result in results.items():
                    conf_matrix = result['conf_matrix']
                    conf_matrix_plot = plot_confusion_matrix(conf_matrix, model_name)
                    st.plotly_chart(conf_matrix_plot)
                    
                    # Tampilkan metrics detail
                    st.write(f"Classification Report untuk {model_name}:")
                    st.text(result['report'])
                
                # Tambahkan visualisasi metrics comparison setelah model dilatih
                st.subheader('Perbandingan Metrics Model:')
                metrics_plot = plot_metrics_comparison(results)
                st.plotly_chart(metrics_plot)
                
                # Plot metrics relationships (fitur sebelumnya)
                st.subheader('Hubungan Antar Metrics:')
                metrics_rel_plot = plot_metrics_relationships(results)
                st.plotly_chart(metrics_rel_plot)
                
                # Tambahkan visualisasi baru
                st.subheader('Distribusi Prediksi:')
                prediction_dist_plot = plot_prediction_distribution(results)
                st.plotly_chart(prediction_dist_plot)
                
                # Bar chart perbandingan akurasi
                st.subheader('Perbandingan Akurasi Model:')
                accuracy_plot = plot_accuracy_comparison(results)
                st.plotly_chart(accuracy_plot)
            else:
                st.warning("Harap lakukan preprocessing data terlebih dahulu!")
        else:
            st.warning("Harap lakukan preprocessing data terlebih dahulu!")

    elif menu == "Prediksi Performa":
        st.header('3. Prediksi Performa')
        if 'results' in st.session_state:
            results = st.session_state['results']
            # Create input form
            with st.form("prediction_form"):
                # Model selection
                model_options = list(results.keys())
                selected_model = st.selectbox('Pilih Model untuk Prediksi:', model_options)
                
                # Input fields
                st.subheader("Data Kategoris")
                categorical_inputs = {}
                col1, col2, col3 = st.columns(3)
                for i, col in enumerate(st.session_state['categorical_columns']):
                    with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                        categorical_inputs[col] = st.selectbox(col, options=st.session_state['label_encoders'][col].classes_)
                
                st.subheader("Data Numerik")
                numerical_inputs = {}
                col1, col2, col3 = st.columns(3)
                for i, col in enumerate(st.session_state['numerical_columns']):
                    with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                        numerical_inputs[col] = st.number_input(col, value=0.0)
                
                submitted = st.form_submit_button("Prediksi")
            
            if submitted:
                # Prepare input data
                input_categorical = pd.DataFrame([categorical_inputs])
                input_numerical = pd.DataFrame([numerical_inputs])
                
                # Transform categorical inputs
                for col in st.session_state['categorical_columns']:
                    input_categorical[col] = st.session_state['label_encoders'][col].transform(input_categorical[col])
                
                # Scale numerical inputs
                input_numerical = pd.DataFrame(
                    st.session_state['scaler'].transform(input_numerical),
                    columns=st.session_state['numerical_columns']
                )
                
                # Combine inputs
                input_data = pd.concat([input_categorical, input_numerical], axis=1)
                
                # Get selected model
                model = st.session_state['results'][selected_model]['model']
                
                # Make prediction
                prediction = model.predict(input_data)
                proba = model.predict_proba(input_data)
                
                # Display result
                st.subheader('Hasil Prediksi:')
                if prediction[0] == 1:
                    st.success('Diprediksi: GRADUATE')
                else:
                    st.error('Diprediksi: DROPOUT/ENROLLED')
                
                # Display prediction probability
                st.write(f"Probabilitas Graduate: {proba[0][1]:.2%}")
                st.write(f"Probabilitas Dropout/Enrolled: {proba[0][0]:.2%}")
                st.write(f"Model yang digunakan: {selected_model}")
                
                # Tampilkan perbandingan dengan model lain
                st.subheader('Perbandingan Prediksi dengan Model Lain:')
                comparison_results = []
                
                for model_name, model_data in st.session_state['results'].items():
                    model_pred = model_data['model'].predict(input_data)
                    model_proba = model_data['model'].predict_proba(input_data)
                    comparison_results.append({
                        'Model': model_name,
                        'Prediksi': 'GRADUATE' if model_pred[0] == 1 else 'DROPOUT/ENROLLED',
                        'Probabilitas Graduate': f"{model_proba[0][1]:.2%}",
                        'Probabilitas Dropout': f"{model_proba[0][0]:.2%}",
                        'Accuracy Model': f"{model_data['accuracy']:.2%}",
                        'F1-Score Model': f"{model_data['f1']:.2%}"
                    })
                
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df.set_index('Model'))
            else:
                st.warning("Harap pilih model dan masukkan data untuk prediksi!")
        else:
            st.warning("Harap train model terlebih dahulu!")

if __name__ == '__main__':
    main()
