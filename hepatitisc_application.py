import pandas as pd
import streamlit as st
import numpy as np
import pickle as pk
import base64


st.set_page_config(
    page_title="Hepatitis C Detection and Prediction System",
    layout="centered"
)


@st.cache_resource
def load_model_and_scaler():
    model = pk.load(open("The_Hepatitis_Model.sav", "rb"))
    scaler = pk.load(open("Hepatitis_saved_std_scaler.pkl", "rb"))
    return model, scaler


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your Predictions</a>'
    return href


def eligibility_status(givendata):
    loaded_model, std_scaler = load_model_and_scaler()

    input_data = np.asarray(givendata, dtype=float).reshape(1, -1)
    std_input_data = std_scaler.transform(input_data)

    prediction = int(loaded_model.predict(std_input_data)[0])

    labels = {
        0: "No hepatitis",
        1: "Suspected Hepatitis",
        2: "Hepatitis is present",
        3: "Fibrosis (mild)",
        4: "Cirrhosis (chronic)"
    }

    return labels.get(prediction, "Unknown prediction")


def get_float_input(label, key):
    value = st.text_input(
        label,
        value="",
        key=key,
        placeholder="Enter value"
    )

    if value.strip() == "":
        return None

    try:
        return float(value)
    except ValueError:
        st.error(f"{label} must be a valid number.")
        return None


def single_prediction():
    st.header("Hepatitis C Detector")

    sex_option = st.selectbox(
        "Select the patient sex",
        ["-- Select sex --", "Male", "Female"],
        key="sex"
    )

    Sex = None
    if sex_option == "Male":
        Sex = 1
    elif sex_option == "Female":
        Sex = 0

    Age = get_float_input("Age", "age")
    ALB = get_float_input("Albumin Blood Test", "alb")
    ALP = get_float_input("Alkaline Phosphatase", "alp")
    ALT = get_float_input("Alanine Transaminase", "alt")
    AST = get_float_input("Aspartate Transaminase", "ast")
    BIL = get_float_input("Bilirubin", "bil")
    CHE = get_float_input("Acetylcholinesterase", "che")
    CHOL = get_float_input("Cholesterol", "chol")
    CREA = get_float_input("Creatinine", "crea")
    GGT = get_float_input("Gamma-Glutamyl Transferase", "ggt")
    PROT = get_float_input("Proteins", "prot")

    inputs = [
        Age, Sex, ALB, ALP, ALT, AST,
        BIL, CHE, CHOL, CREA, GGT, PROT
    ]

    form_complete = all(value is not None for value in inputs)

    if not form_complete:
        st.warning("Please fill in all fields before prediction.")

    predict_button = st.button(
        "Predict",
        disabled=not form_complete
    )

    if predict_button:
        result = eligibility_status(inputs)
        st.success(result)


def multi_prediction(input_data):
    loaded_model, std_scaler = load_model_and_scaler()

    dfinput = pd.read_csv(input_data)

    st.header("Preview of the Uploaded Dataset")
    st.dataframe(dfinput)

    required_columns = [
        "Age", "Sex", "ALB", "ALP", "ALT", "AST",
        "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
    ]

    if list(dfinput.columns) != required_columns:
        st.error("CSV columns must be in this exact order:")
        st.write(required_columns)
        return

    if dfinput.isnull().values.any():
        st.error("The uploaded CSV contains missing values. Please clean the file and upload again.")
        return

    predict_button = st.button("Predict")

    if predict_button:
        scaled_data = std_scaler.transform(dfinput)
        predictions = loaded_model.predict(scaled_data)

        labels = {
            0: "No hepatitis",
            1: "Suspected Hepatitis",
            2: "Hepatitis is present",
            3: "Fibrosis (mild)",
            4: "Cirrhosis (chronic)"
        }

        prediction_labels = [labels[int(pred)] for pred in predictions]

        dfresult = pd.DataFrame({
            "S/N": np.arange(1, len(prediction_labels) + 1),
            "Category": prediction_labels
        })

        st.subheader("Predicted Output")
        st.dataframe(dfresult)

        st.markdown(filedownload(dfresult), unsafe_allow_html=True)


with st.sidebar:
    selection = st.radio(
        "Choose your prediction system",
        ["Single Prediction", "Multi Prediction"]
    )


if selection == "Single Prediction":
    single_prediction()

elif selection == "Multi Prediction":
    st.header("Multi Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        multi_prediction(uploaded_file)
    else:
        st.info("Waiting for CSV file to be uploaded.")
