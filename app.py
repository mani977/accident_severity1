import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import ordinal_encoder,get_prediction
model=joblib.load(r"model.joblib")
st.set_page_config(page_title="Accident Severity Prediction",layout="wide")

sex= ['Male' ,'Female', 'Unknown']
#age_of_driver=['18-30','31-50','Over 51','Unknown','Under 18']
Weather_conditions 	= ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow',
 'Unknown', 'Fog or mist']
time=[]
road_surface_type=['Asphalt roads', 'Earth roads', 11296,
       'Asphalt roads with some distress', 'Gravel roads', 'Other']
st.markdown("<h1 style='text-align:center;'>Accident severity prediction applicaiton </h1>",unsafe_allow_html=True)
def main():
    with st.form("prediction_form"):
       st.subheader("enter the following features")
       time=st.slider("pickup hour",0,23,value=0,format="%d")
       whether=st.selectbox("select whether",options=Weather_conditions)
       driver_age=st.number_input("enter age")
       driver_age=int(driver_age)
       sex_of_driver=st.selectbox("select gender",options=sex)
       road_surface=st.selectbox("select surface",options=road_surface_type)
       submit=st.form_submit_button("predict")
    if submit:
        whether=ordinal_encoder(whether,Weather_conditions)
        #driver_age=ordinal_encoder(driver_age,age_of_driver)
        sex_of_driver=ordinal_encoder(sex_of_driver,sex)
        road_surface=ordinal_encoder(road_surface,road_surface_type)

        data=np.array([driver_age, sex_of_driver, whether, time,road_surface]).reshape(1,-1)
        pred=get_prediction(data=data,model=model)
        st.write(f"the predicted severitu is {pred[0]}")
if __name__=="__main__":
    main()
                  



   


        

