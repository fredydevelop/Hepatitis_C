import pandas as pd
import streamlit as st
import numpy as np
import pickle as pk
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report 
import base64
from sklearn import svm
#import seaborn as sns
import altair as alt
from sklearn.preprocessing import StandardScaler




st.set_page_config(page_title='Herpatitis detection and prediction system',layout='centered')



# with st.sidebar:
#     #selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction","Model Performance"],icons=["cast","book","cast"],menu_icon="house",default_index=0)

#     selection = st.radio(
#     "Choose your prediction system",
#     ["Single Prediction", "Multi Prediction"])

#     if selection == 'Single Prediction':
#         main()
#     else:
#         st.set_option('deprecation.showPyplotGlobalUse', False)
#         st.title("predicto")
#         uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
#         #--------------Visualization-------------------#
#         # Main panel
        
#         # Displays the dataset
#         if uploaded_file is not None:
#             #load_data = pd.read_table(uploaded_file)
#             multi(uploaded_file)
#         else:
#             st.info('waiting for CSV file to be uploaded.')




# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your Predictions</a>'
    return href


def eligibility_status(givendata):
    
    loaded_model=pk.load(open("The_Hepatitis_Model.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    std_scaler=pk.load(open("Hepatitis_saved_std_scaler.pkl", "rb"))
    std_input_data_reshaped=std_scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_input_data_reshaped)
    if prediction==0:
      return"No hepatitis"
    if prediction==1:
        return "Suspected Hepatitis"
    if prediction==2:
        return "Hepatitis is present"
    if prediction==3:
        return "Fibrosis(mild)"
    if prediction==4:
      return "Cirrhosis(chronic)"



def main():
    st.header("Herpatitis C Detector")
    theReasons=[]


    
    #getting user input
    option = st.radio("select the patient sex",["Male",'Female'],key="gender")
    if (option=='Male'):
        Sex=1
    else:
        Sex=0
    st.write("\n")
    st.write("\n")

    # if Service==0:
    #     theReasons.append("Postpaid users are not eligible for a loan")
    
    Age =  st.slider('How old are you?',0 ,130,key="age")
    st.write("Patient is",Age,"years old")
    st.write("\n")
    st.write("\n")

    try:
        ALB =  st.text_input('Albumin Blood Test',"")
        ALB=float(ALB)
    except:
        st.error("Incorrect Input")

    try:
        ALP =  st.text_input('Alkaline phosphatase',"0",key="alp")
        ALP=float(ALP)
    except:
        st.error("Incorrect Input")


    try:
        # ALP =  st.text_input('Movie title',"0",key="alp")
        ALT =  st.text_input('Alanine Transaminase',"0",key="alt")
        ALT=float(ALT)
    except:
        st.error("Incorrect Input")

    try:

        AST =  st.text_input('Aspartate Transaminase',"0",key="ast")
        AST=float(AST)
    except:
        st.error("Incorrect Input")

    try:

        BIL =  st.text_input('Bilirubin',"0",key="bil")
        BIL=float(BIL)
    except TypeError:
        st.error("Incorrect Input")

    try:
        CHE =  st.text_input('Acetylcholinesterase',"0",key="che")
        CHE=float(CHE)
    except:
        st.error("Incorrect Input")

    try:

        CHOL =  st.text_input('Cholesterol',"0",key="chol")
        CHOL=float(CHOL)
    except:
        st.error("Incorrect Input")

    try:

        CREA =  st.text_input('Creatinine',"0",key="crea")
        CREA=float(CREA)
    except:
        st.error("Incorrect Input")
    
    try:   
        GGT =  st.text_input('Gamma-Glutamyl Transferase',"0",key="ggt")
        GGT=float(GGT)
    except:
        st.error("Incorrect Input")

    try:
        PROT =  st.text_input('Proteins',"0",key="prot")
        PROT=float(PROT)
    except:
        st.error("Incorrect Input")

   
    Eligible = '' #for displaying result
    medical_advice=""
    
    # creating a button for Prediction
    if option!=None and Age!=None and ALB!=None and ALP!=None and ALT!=None and AST!=None and BIL!=None and CHE!=None and CHOL!=None and CREA!=None and GGT!=None and PROT!=None and st.button('Predict'):
        Eligible = eligibility_status([Age, Sex, ALB, ALP, ALT,AST,BIL, CHE, CHOL, CREA, GGT, PROT])
        st.success(Eligible)
    
    
    if Eligible=="No hepatitis":
      advice=["Maintain a balanced diet rich in fruits, vegetables, and whole grains to support liver health.",
    "Exercise regularly and avoid excessive alcohol consumption to keep your liver in optimal condition.",
    "Remember to schedule regular check-ups with your healthcare provider to monitor your liver function."]
      st.write("Medical Advice")
      j=0
      k=1
      while j< len(advice):
        st.write(str(k) + ". " + advice[j])
        j=j+1
        k=k+1

    elif Eligible=="Suspected Hepatitis":
        advice=["It's important to avoid sharing personal items like razors or toothbrushes to prevent potential transmission.","Consider consulting a healthcare professional for further testing to confirm or rule out hepatitis.","In the meantime, practice good hygiene and take precautions to protect others from potential exposure."]
        st.write("Medical Advice")
        j=0
        k=1
        while j< len(advice):
            st.write(str(k) + ". " + advice[j])
            j=j+1
            k=k+1
    elif Eligible=="Hepatitis is present":
        advice=["You've been diagnosed with Hepatitis C. It's crucial to follow your healthcare provider's treatment plan.",
    "Discuss treatment options, potential side effects, and any concerns you may have with your healthcare team.",
    "Avoid alcohol and certain medications that may further strain your liver. Follow a liver-friendly diet."]
        st.write("Medical Advice")
        j=0
        k=1
        while j< len(advice):
            st.write(str(k) + ". " + advice[j])
            j=j+1
            k=k+1 
    elif Eligible=="Fibrosis(mild)":
        advice=["Fibrosis is an early stage of liver scarring. Focus on a low-fat, high-fiber diet to support your liver.",
    "Regular exercise can help improve liver function. Consult your healthcare provider for personalized advice.",
    "Stay on top of follow-up appointments to monitor the progression of fibrosis and discuss any concerns."]
        st.write("Medical Advice")
        j=0
        k=1
        while j< len(advice):
            st.write(str(k) + ". " + advice[j])
            j=j+1
            k=k+1 
    elif Eligible=="Cirrhosis(chronic)" :
        advice=["Managing cirrhosis requires close medical supervision. Follow your healthcare provider's recommendations diligently.",
    "Limit sodium intake to reduce fluid retention. Your provider can guide you on an appropriate diet plan.",
    "Keep a record of any new symptoms and report them to your healthcare team promptly."]
        st.write("Medical Advice")
        j=0
        k=1
        while j< len(advice):
            st.write(str(k) + ". " + advice[j])
            j=j+1
            k=k+1 
    else:
        st.write("")
    #     st.balloons()

 
    

interchange=[]


def multi(input_data):
    loaded_model=pk.load(open("The_Hepatitis_Model.sav", "rb"))
    dfinput = pd.read_csv(input_data)
  
    st.header('Preview of the uploaded dataset')
    # st.markdown('Preview of the uploaded dataset')
    st.dataframe(dfinput)
    #st.markdown(dfinput.shape)
    st.write("\n")
    st.write("\n")
    st.write("\n")
    # dfinput=pd.DataFrame(dfinput.values)
    #dfinput=dfinput.iloc[1:].reset_index(drop=True)
    

    std_scaler=pk.load(open("Hepatitis_saved_std_scaler.pkl", "rb"))
    dfinput=std_scaler.transform(dfinput)


    with st.sidebar:
        
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        predictButton=st.button("Predict")



    if predictButton:
    #     st.write("Standardized dataset")
    #     st.dataframe(dfinput)
        prediction = loaded_model.predict(dfinput)
        for i in prediction:
            if i==0:
                interchange.append("No hepatitis")
            elif i==1:
                interchange.append("Suspected Hepatitis")
            elif i==2:
                interchange.append("Hepatitis is present")
                    
            elif i==3:
                interchange.append("Fibrosis(mild)")
                    
            else:
                interchange.append("Cirrhosis(chronic)")
                
                
        st.subheader('**Predicted output**')
        prediction_output = pd.Series(interchange, name='Category')
        prediction_id = pd.Series(np.arange(0,len(interchange)))
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

        

with st.sidebar:
    #selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction","Model Performance"],icons=["cast","book","cast"],menu_icon="house",default_index=0)
    st.image("Hepatitisclogo.png",width=250)
    selection = st.radio(
    "Choose your prediction system",
    ["Single Prediction", "Multi Prediction"])

        #--------------Visualization-------------------#
        # Main panel
        
        # Displays the dataset
        
if selection == 'Multi Prediction':

    st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.title("predicto")
    uploaded_file = st.file_uploader("", type=["csv"])
    if uploaded_file is not None:
                #load_data = pd.read_table(uploaded_file)
        multi(uploaded_file)
    else:
        st.info('waiting for CSV file to be uploaded.')

if selection == 'Single Prediction':
    main()

