import streamlit as st
import pandas as pd
import pickle

# Load model
def load_model(model_name):
    if model_name == 'Logistic Regression':
        model = pickle.load(open('models/no_resampling_logistic_regression_model.pkl', 'rb'))
    elif model_name == 'SVM':
        model = pickle.load(open('models/no_resampling_svm_model.pkl', 'rb'))
    # elif model_name == 'Decission Tree':
    #     model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))
    # elif model_name == 'Random Forest':
    #     model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
    # elif model_name == 'XGB':
    #     model = pickle.load(open('models/xgboost_model.pkl', 'rb'))
    # elif model_name == 'GBM':
    #     model = pickle.load(open('models/gradient_boosting_model.pkl', 'rb'))
    return model

# Fungsi untuk melakukan prediksi
def predict_attrition(model, data):
    predictions = model.predict(data)
    return predictions

# Fungsi untuk mewarnai prediksi
def color_predictions(val):
    color = 'red' if val == 'Dropout' else 'green'
    return f'color: {color}'

def main():
    st.title('Prediksi Student Performance menggunakan Model Machine Learning')

    # Sidebar
    st.sidebar.title("About Me")
    st.sidebar.write("Name            : Maulana Kavaldo")
    st.sidebar.markdown("Id Dicoding\t: [maulanakavaldo](https://www.dicoding.com/users/maulanakavaldo/).")
    st.sidebar.markdown("Reach out me : [LinkedIn](https://www.linkedin.com/in/maulana-kavaldo/).")

    with st.expander("Petunjuk Penggunaan:"):
        st.write(
            """
                1. Buka sidebar, disebelah kiri terlihat tanda panah. Kamu bisa men-kliknya sehingga sidebar dapat terbuka.
                2. Unggah file CSV yang berisi data karyawan. Kamu bisa menggunakan contoh sample_test.csv yang terdapat pada proyek ini.
                3. Pilih model machine learning.
                4. Klik tombol "Prediksi' untuk melihat hasil prediksi.
                5. Kamu bisa melakukan download hasil prediksi dengan klik tombol 'Download (.csv)'
            """
        )

    # Pemilihan model ML
    model_name = st.sidebar.selectbox("Pilih Model Machine Learning", ("Logistic Regression", "SVM"
                                                                        # ,"Decission Tree", "Random Forest", "XGB", "GBM"
                                                                       ))

    # Upload File
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV untuk prediksi", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("Data test:")
        preview_rows = st.slider("Kamu mau melihat berapa data? Coba gerakkan slider ini ya.", 1, len(data), 5)
        st.write(data.head(preview_rows))

        # Extract StudentId dan StudentName
        ID = data['ID']
        StudentName = data['Name']
        data = data.drop(columns=['ID', 'Name'])

        # Load model yang dipilih
        model = load_model(model_name)

        # Button untuk trigger
        if st.button('Prediksi'):
            # Melakukan prediksi
            predictions = predict_attrition(model, data)

            # Mengubah value agar mudah dipahami
            prediction_labels = ['Graduate' if pred == 1 else 'Dropout' for pred in predictions]

            # Menampilkan hasilnya
            result_df = pd.DataFrame({
                'ID': ID,
                'Name': StudentName,
                'Status Prediction': prediction_labels
            })

            # Menampilkan hasil prediksi dengan styling
            st.write("Hasil Prediksi:")
            st.dataframe(result_df.style.applymap(color_predictions, subset=['Status Prediction']))

            # Download hasil prediksi
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download (.csv)",
                data=csv,
                file_name='hasil-prediksi-student.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()
