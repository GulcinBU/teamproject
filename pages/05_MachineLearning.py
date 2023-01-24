import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


#logo
def add_logo(logo_url: str):
    """Add a logo (from logo_url) on the top of the navigation page of a multipage app.
    Taken from https://discuss.streamlit.io/t/put-logo-and-title-above-on-top-of-page-navigation-in-sidebar-of-multipage-app/28213/6

    Args:
        logo_url (HttpUrl): URL of the logo
    """



    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url({logo_url});
                background-repeat: no-repeat;
                padding-top: 220px;
                background-position: 0px 0px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgHYxaZGEwO_WB0G1XNt8YIRBROxYZe4Gu1xsQv-am-nNMEYwxhPhMGnTiIHsU_e3ginb1UcDlAy68jOoHN16Lh3088VutGmidRx0rv1OWdDkVsIGCM8RoGY-V8sZ7yUEiTsMoAWESPts6qOyZsfeH0q4EOqL0fVg6GlEB6sqGP2OKZeceEMJzv_Y4bRA/s320/logo.png")

st.title('Machine Learning')
st.write('Please upload your cleaned data which you downloaded in the "Upload Data" section.')

#upload cleaned file
uploaded_files = st.file_uploader("Choose  CSV file(s)", accept_multiple_files=True, type=['csv'])
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        columnheaders = df.columns.tolist()
        y_var = st.selectbox('Select the variable to be predicted (y)', columnheaders)
        X_var = st.multiselect('Select the variables to be used for prediction (X)', columnheaders)
        if len(X_var) == 0:
            st.error("You have to put in some X variable and it cannot be left empty.")

            # Check if y not in X
        if y_var in X_var:
            st.error("Warning! Y variable cannot be present in your X-variable.")

    # Option to select predition type

        pred_type = st.radio("Select the type of process you want to run.",
                             options=['Linear regression', 'Logistic regression', 'KNN', 'Gaussian Naive Bayes', 'SVM', 'Decision tree',
                  'Random forest'],
                             help="Write about reg and classification")
        params = {
            'X': X_var,
            'y': y_var,
            'pred_type': pred_type,
        }
        if st.button("Run Models"):
            st.write(f"**Variable to be predicted:** {y_var}")
            st.write(f"**Variable to be used for prediction:** {X_var}")

            # Divide the data into test and train set
        X = df[X_var]
        y = df[y_var]

        # Perform encoding
        X = pd.get_dummies(X)


        def isNumerical(col):
            return is_numeric_dtype(col)

        # Check if y needs to be encoded
        if not isNumerical(y):
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Print all the classes
            st.write("The classes and the class allotted to them is the following:-")
            classes = list(le.classes_)
            for i in range(len(classes)):
                st.write(f"{classes[i]} --> {i}")

        ## Split data in test and train
        st.markdown("#### Train Test Splitting")
        trainingsize = st.slider('Select training size', min_value=0.0, max_value=1.0, value=0.7, help="This is the value which will be used to divide the data for training and testing. Default = 70%")


        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainingsize, random_state=0)
        st.write("Number of training samples:", X_train.shape[0])
        st.write("Number of testing samples:", X_test.shape[0])


        if pred_type == "Linear regression":
            st.subheader("Lineer Regression Results")
            model_lin = LinearRegression()
            model_lin.fit(X_train, y_train)
            y_pred_lin = model_lin.predict(X_test)
            try:
                confusion_matrix_lin = confusion_matrix(y_test, y_pred_lin)
                classification_report_lin = classification_report(y_test, y_pred_lin)
                accuracy_score_lin = accuracy_score(y_test, y_pred_lin)
                st.write("Confusion Matrix:", confusion_matrix_lin)
                st.write("Classification Report:", classification_report_lin)
                st.write("Accuracy Score:", accuracy_score_lin)

            except ValueError:
                rmse_lin = np.sqrt(np.mean((y_test - y_pred_lin) ** 2))
                mae_lin = mean_absolute_error(y_test, y_pred_lin)
                st.write("Root Mean Squared Error:", rmse_lin)
                st.write("Mean Absolute Error:", mae_lin)

        if pred_type == "Logistic regression":
            st.subheader("Logistic Regression Results")
            model_log = LogisticRegression()
            model_log.fit(X_train, y_train)
            y_pred_log = model_log.predict(X_test)
            try:
                confusion_matrix_log = confusion_matrix(y_test, y_pred_log)
                classification_report_log = classification_report(y_test, y_pred_log)
                accuracy_score_log = accuracy_score(y_test, y_pred_log)
                st.write("Confusion Matrix:", confusion_matrix_log)
                st.write("Classification Report:", classification_report_log)
                st.write("Accuracy Score:", accuracy_score_log)
            except ValueError:
                rmse_log = np.sqrt(np.mean((y_test - y_pred_log) ** 2))
                mae_log = mean_absolute_error(y_test, y_pred_log)
                st.write("Root Mean Squared Error:", rmse_log)
                st.write("Mean Absolute Error:", mae_log)

        if pred_type == "KNN":
            st.subheader("KNN Results")
            model_knn = KNeighborsClassifier(n_neighbors=3)
            model_knn.fit(X_train, y_train)
            y_pred_knn = model_knn.predict(X_test)
            try:
                confusion_matrix_knn = confusion_matrix(y_test, y_pred_knn)
                classification_report_knn = classification_report(y_test, y_pred_knn)
                accuracy_score_knn = accuracy_score(y_test, y_pred_knn)
                st.write("Confusion Matrix:", confusion_matrix_knn)
                st.write("Classification Report:", classification_report_knn)
                st.write("Accuracy Score:", accuracy_score_knn)
            except ValueError:
                rmse_knn = np.sqrt(np.mean((y_test - y_pred_knn) ** 2))
                mae_knn = mean_absolute_error(y_test, y_pred_knn)
                st.write("Root Mean Squared Error:", rmse_knn)
                st.write("Mean Absolute Error:", mae_knn)

        if pred_type == "Gaussian Naive Bayes ":
            st.subheader("Gaussian Naive Bayes Results")
            model_nb = GaussianNB()
            model_nb.fit(X_train, y_train)
            y_pred_nb = model_nb.predict(X_test)
            try:
                confusion_matrix_nb = confusion_matrix(y_test, y_pred_nb)
                classification_report_nb = classification_report(y_test, y_pred_nb)
                accuracy_score_nb = accuracy_score(y_test, y_pred_nb)
                st.write("Confusion Matrix:", confusion_matrix_nb)
                st.write("Classification Report:", classification_report_nb)
                st.write("Accuracy Score:", accuracy_score_nb)
            except ValueError:
                rmse_nb = np.sqrt(np.mean((y_test - y_pred_nb) ** 2))
                mae_nb = mean_absolute_error(y_test, y_pred_nb)
                st.write("Root Mean Squared Error:", rmse_nb)
                st.write("Mean Absolute Error:", mae_nb)

        if pred_type == "SVM":
            st.subheader("SVM Results")
            model_svc = SVC(gamma='scale')
            model_svc.fit(X_train, y_train)
            y_pred_svc = model_svc.predict(X_test)
            try:
                confusion_matrix_svc = confusion_matrix(y_test, y_pred_svc)
                classification_report_svc = classification_report(y_test, y_pred_svc)
                accuracy_score_svc = accuracy_score(y_test, y_pred_svc)
                st.write("Confusion Matrix:", confusion_matrix_svc)
                st.write("Classification Report:", classification_report_svc)
                st.write("Accuracy Score:", accuracy_score_svc)
            except ValueError:
                rmse_svc = np.sqrt(np.mean((y_test - y_pred_svc) ** 2))
                mae_svc = mean_absolute_error(y_test, y_pred_svc)
                st.write("Root Mean Squared Error:", rmse_svc)
                st.write("Mean Absolute Error:", mae_svc)

        if pred_type == "Decision tree":
            st.subheader("Decision Tree Results")
            model_tree = DecisionTreeClassifier()
            model_tree.fit(X_train, y_train)
            y_pred_tree = model_tree.predict(X_test)
            try:
                confusion_matrix_tree = confusion_matrix(y_test, y_pred_tree)
                classification_report_tree = classification_report(y_test, y_pred_tree)
                accuracy_score_tree = accuracy_score(y_test, y_pred_tree)
                st.write("Confusion Matrix:", confusion_matrix_tree)
                st.write("Classification Report:", classification_report_tree)
                st.write("Accuracy Score:", accuracy_score_tree)
            except ValueError:
                rmse_tree = np.sqrt(np.mean((y_test - y_pred_tree) ** 2))
                mae_tree = mean_absolute_error(y_test, y_pred_tree)
                st.write("Root Mean Squared Error:", rmse_tree)
                st.write("Mean Absolute Error:", mae_tree)

        if pred_type == "Random forest":
            st.subheader("Random forest Results")
            model_forest = RandomForestClassifier()
            model_forest.fit(X_train, y_train)
            y_pred_forest = model_forest.predict(X_test)
            try:
                confusion_matrix_forest = confusion_matrix(y_test, y_pred_forest)
                classification_report_forest = classification_report(y_test, y_pred_forest)
                accuracy_score_forest = accuracy_score(y_test, y_pred_forest)
                st.write("Confusion Matrix:", confusion_matrix_forest)
                st.write("Classification Report:", classification_report_forest)
                st.write("Accuracy Score:", accuracy_score_forest)
            except ValueError:
                rmse_forest = np.sqrt(np.mean((y_test - y_pred_forest) ** 2))
                mae_forest = mean_absolute_error(y_test, y_pred_forest)
                st.write("Root Mean Squared Error:", rmse_forest)
                st.write("Mean Absolute Error:", mae_forest)








