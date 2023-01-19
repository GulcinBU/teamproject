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

# st.title('Machine Learning')
# st.write('Please upload your cleaned data which you downloaded in the "Upload Data" section.')
#
# data = pd.read_csv("file (1).csv")
# df = pd.DataFrame(data)
# columnheaders = df.columns.tolist()
# pred_target = st.selectbox('Select prediction target', columnheaders)
# features =st.multiselect('Select features', columnheaders)
# trainingsize = st.slider('Select training size', min_value=0.0, max_value=1.0, value=0.7)
# ## Split data in test and train
# y = df[pred_target]
# X = features
# for i in features:
#     List =[i]
#     st.write(List)
#     for n in columnheaders:
#         n !=i
#         df1 = df.drop(df[n])
#         st.write(df1)

## Testing ML models
data = pd.read_csv("file (1).csv")
df = pd.DataFrame(data)

dummy = pd.get_dummies(df, dummy_na=True)
X = dummy.drop('selling_price', axis=1)
y = dummy['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

model_lin = LinearRegression()
model_lin.fit(X_train, y_train)
y_pred_lin = model_lin.predict(X_test)
try:
    confusion_matrix_lin = confusion_matrix(y_test, y_pred_lin)
    classification_report_lin = classification_report(y_test, y_pred_lin)
    accuracy_score_lin = accuracy_score(y_test, y_pred_lin)
    print("Confusion Matrix:", confusion_matrix_lin)
    print("Classification Report:", classification_report_lin)
    print("Accuracy Score:", accuracy_score_lin)
except ValueError:
    rmse_lin = np.sqrt(np.mean((y_test - y_pred_lin) ** 2))
    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    print("Root Mean Squared Error:", rmse_lin)
    print("Mean Absolute Error:", mae_lin)

model_log = LogisticRegression()
model_log.fit(X_train,y_train)
y_pred_log = model_log.predict(X_test)
try:
    confusion_matrix_log = confusion_matrix(y_test, y_pred_log)
    classification_report_log = classification_report(y_test, y_pred_log)
    accuracy_score_log = accuracy_score(y_test, y_pred_log)
    print("Confusion Matrix:", confusion_matrix_log)
    print("Classification Report:", classification_report_log)
    print("Accuracy Score:", accuracy_score_log)
except ValueError:
    rmse_log = np.sqrt(np.mean((y_test - y_pred_log)**2))
    mae_log = mean_absolute_error(y_test,y_pred_log)
    print("Root Mean Squared Error:", rmse_log)
    print("Mean Absolute Error:", mae_log)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train,y_train)
y_pred_knn = model_knn.predict(X_test)
try:
    confusion_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    classification_report_knn = classification_report(y_test, y_pred_knn)
    accuracy_score_knn = accuracy_score(y_test, y_pred_knn)
    print("Confusion Matrix:", confusion_matrix_knn)
    print("Classification Report:", classification_report_knn)
    print("Accuracy Score:", accuracy_score_knn)
except ValueError:
    rmse_knn = np.sqrt(np.mean((y_test - y_pred_knn)**2))
    mae_knn = mean_absolute_error(y_test,y_pred_knn)
    print("Root Mean Squared Error:", rmse_knn)
    print("Mean Absolute Error:", mae_knn)

model_nb = GaussianNB()
model_nb.fit(X_train,y_train)
y_pred_nb = model_nb.predict(X_test)
try:
    confusion_matrix_nb = confusion_matrix(y_test, y_pred_nb)
    classification_report_nb = classification_report(y_test, y_pred_nb)
    accuracy_score_nb = accuracy_score(y_test, y_pred_nb)
    print("Confusion Matrix:", confusion_matrix_nb)
    print("Classification Report:", classification_report_nb)
    print("Accuracy Score:", accuracy_score_nb)
except ValueError:
    rmse_nb = np.sqrt(np.mean((y_test - y_pred_nb)**2))
    mae_nb = mean_absolute_error(y_test,y_pred_nb)
    print("Root Mean Squared Error:", rmse_nb)
    print("Mean Absolute Error:", mae_nb)

model_svc = SVC(gamma = 'scale')
model_svc.fit(X_train,y_train)
y_pred_svc = model_svc.predict(X_test)
try:
    confusion_matrix_svc = confusion_matrix(y_test, y_pred_svc)
    classification_report_svc = classification_report(y_test, y_pred_svc)
    accuracy_score_svc = accuracy_score(y_test, y_pred_svc)
    print("Confusion Matrix:", confusion_matrix_svc)
    print("Classification Report:", classification_report_svc)
    print("Accuracy Score:", accuracy_score_svc)
except ValueError:
    rmse_svc = np.sqrt(np.mean((y_test - y_pred_svc)**2))
    mae_svc = mean_absolute_error(y_test,y_pred_svc)
    print("Root Mean Squared Error:", rmse_svc)
    print("Mean Absolute Error:", mae_svc)

model_tree = DecisionTreeClassifier()
model_tree.fit(X_train,y_train)
y_pred_tree =  model_tree.predict(X_test)
try:
    confusion_matrix_tree = confusion_matrix(y_test, y_pred_tree)
    classification_report_tree = classification_report(y_test, y_pred_tree)
    accuracy_score_tree = accuracy_score(y_test, y_pred_tree)
    print("Confusion Matrix:", confusion_matrix_tree)
    print("Classification Report:", classification_report_tree)
    print("Accuracy Score:", accuracy_score_tree)
except ValueError:
    rmse_tree = np.sqrt(np.mean((y_test - y_pred_tree)**2))
    mae_tree = mean_absolute_error(y_test,y_pred_tree)
    print("Root Mean Squared Error:", rmse_tree)
    print("Mean Absolute Error:", mae_tree)

model_forest = RandomForestClassifier()
model_forest.fit(X_train,y_train)
y_pred_forest = model_forest.predict(X_test)
try:
    confusion_matrix_forest = confusion_matrix(y_test, y_pred_forest)
    classification_report_forest = classification_report(y_test, y_pred_forest)
    accuracy_score_forest = accuracy_score(y_test, y_pred_forest)
    print("Confusion Matrix:", confusion_matrix_forest)
    print("Classification Report:", classification_report_forest)
    print("Accuracy Score:", accuracy_score_forest)
except ValueError:
    rmse_forest = np.sqrt(np.mean((y_test - y_pred_forest)**2))
    mae_forest = mean_absolute_error(y_test,y_pred_forest)
    print("Root Mean Squared Error:", rmse_forest)
    print("Mean Absolute Error:", mae_forest)

#Overview outcome
st.dataframe(pd.DataFrame({
'Metrics': ['Confusion matrix', 'Classification report', 'Accuracy score', 'RMSE', 'MAE'],
'Linear regression': [confusion_matrix_lin, classification_report_lin, accuracy_score_lin, rmse_lin, mae_lin],
'Logistic regression': [confusion_matrix_log, classification_report_log, accuracy_score_log, rmse_log, mae_log],
'KNN':[confusion_matrix_knn, classification_report_knn, accuracy_score_knn, rmse_knn, mae_knn],
'Gaussian Naive Bayes':[confusion_matrix_nb, classification_report_nb, accuracy_score_nb, rmse_nb, mae_nb],
'Support Vector Machines':[confusion_matrix_svc, classification_report_svc, accuracy_score_svc, rmse_svc, mae_svc],
'Decision trees':[confusion_matrix_tree, classification_report_tree, accuracy_score_tree, rmse_tree, mae_tree],
'Random forest':[confusion_matrix_forest, classification_report_forest, accuracy_score_forest, rmse_forest, mae_forest],
}))
