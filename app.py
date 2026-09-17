import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Surivival Chance in the Titanic Journey")

plcass=st.slider('Enter the Passenger Class for the user',1,3)

sex= st.selectbox('Enter the Passenger Gender',['male','female'])

sibsp= st.slider('Enter the Passenger total Sibling and Spouse',1,8)

parch= st.slider('Enter the Passenger total number of parents and Child',0,6)

fare= st.number_input('Enter the Fare of the Passenger')

embarked=st.selectbox('Enter the Passenger station from where they started the Journey',['Southampton','chebourg','Queenstown'])


data= pd.DataFrame([{
    'Pclass': plcass,
    'Sex': sex,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}])

#if st.button('Data'):
#    st.write(data)

model=load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

data['Sex']=label.transform(data['Sex'])

embarked=onehot.transform(data[['Embarked']])

embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked], axis=1)

num_cols=['Pclass', 'SibSp', 'Parch', 'Fare']

data[num_cols]=scaler.transform(data[num_cols])


y= model.predict(data)

y=y[0][0]

def chance(y):
    if y>0.5:
        return 'The Passenger will Survive the Journey'
    else:
        return "The Passenger won't survive the Journey"
    
if st.button('Predict Survival Chance'):
    st.write('Probablity of Passenger Survival Chance',y)
    st.write(chance(y))
