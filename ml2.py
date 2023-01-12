import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error
import numpy as np

st.title('Machine Learning')
st.write('Please upload your cleaned data which you downloaded in the "Upload Data" section.')

data = pd.read_csv("file (1).csv")
df = pd.DataFrame(data)
columnheaders = df.columns.tolist()
pred_target = st.selectbox('Select prediction target', columnheaders)
features =st.multiselect('Select features', columnheaders)
trainingsize = st.slider('Select training size', min_value=0.0, max_value=1.0, value=0.7)
## Split data in test and train
y = df[pred_target]
for i in features:
    List =[i]
    st.write(List)
    for n in columnheaders:
        n !=i
        df1 = df.drop(df[n])
        st.write(df1)











