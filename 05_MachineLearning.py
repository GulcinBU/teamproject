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
        pred_target = st.selectbox('Select prediction target', columnheaders)
        trainingsize = st.slider('Select training size', min_value=0.0, max_value=1.0, value=0.7)
        ## Split data in test and train
        dummy = pd.get_dummies(df, dummy_na=True)
        y = dummy[pred_target]
        X = dummy.drop(pred_target, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainingsize, random_state=0)

        option = ['Linear regression', 'Logistic regression', 'KNN', 'Gaussian Naive Bayes', 'SVM', 'Decision tree',
                  'Random forest']
        selected_options = st.multiselect('Select the model(s) you want to use', option)

        if option == 'Linear regression' in selected_options:
            model_lin = LinearRegression()
            model_lin.fit(X_train, y_train)
            y_pred_lin = model_lin.predict(X_test)
            try:
                confusion_matrix_lin = confusion_matrix(y_test, y_pred_lin)
                classification_report_lin = classification_report(y_test, y_pred_lin)
                accuracy_score_lin = accuracy_score(y_test, y_pred_lin)
                st.success("Confusion Matrix:", confusion_matrix_lin)
                st.success("Classification Report:", classification_report_lin)
                st.success("Accuracy Score:", accuracy_score_lin)
            except ValueError:
                rmse_lin = np.sqrt(np.mean((y_test - y_pred_lin) ** 2))
                mae_lin = mean_absolute_error(y_test, y_pred_lin)
                st.success("Root Mean Squared Error:", rmse_lin)
                st.success("Mean Absolute Error:", mae_lin)

        if option == 'Logistic regression' in selected_options:
            model_log = LogisticRegression()
            model_log.fit(X_train, y_train)
            y_pred_log = model_log.predict(X_test)
            try:
                confusion_matrix_log = confusion_matrix(y_test, y_pred_log)
                classification_report_log = classification_report(y_test, y_pred_log)
                accuracy_score_log = accuracy_score(y_test, y_pred_log)
                st.success("Confusion Matrix:", confusion_matrix_log)
                st.success("Classification Report:", classification_report_log)
                st.success("Accuracy Score:", accuracy_score_log)
            except ValueError:
                rmse_log = np.sqrt(np.mean((y_test - y_pred_log) ** 2))
                mae_log = mean_absolute_error(y_test, y_pred_log)
                st.success("Root Mean Squared Error:", rmse_log)
                st.success("Mean Absolute Error:", mae_log)

        if option == 'KNN' in selected_options:
            model_knn = KNeighborsClassifier(n_neighbors=3)
            model_knn.fit(X_train, y_train)
            y_pred_knn = model_knn.predict(X_test)
            try:
                confusion_matrix_knn = confusion_matrix(y_test, y_pred_knn)
                classification_report_knn = classification_report(y_test, y_pred_knn)
                accuracy_score_knn = accuracy_score(y_test, y_pred_knn)
                st.success("Confusion Matrix:", confusion_matrix_knn)
                st.success("Classification Report:", classification_report_knn)
                st.success("Accuracy Score:", accuracy_score_knn)
            except ValueError:
                rmse_knn = np.sqrt(np.mean((y_test - y_pred_knn) ** 2))
                mae_knn = mean_absolute_error(y_test, y_pred_knn)
                st.success("Root Mean Squared Error:", rmse_knn)
                st.success("Mean Absolute Error:", mae_knn)

        if option == 'Gaussian Naive Bayes' in selected_options:
            model_nb = GaussianNB()
            model_nb.fit(X_train, y_train)
            y_pred_nb = model_nb.predict(X_test)
            try:
                confusion_matrix_nb = confusion_matrix(y_test, y_pred_nb)
                classification_report_nb = classification_report(y_test, y_pred_nb)
                accuracy_score_nb = accuracy_score(y_test, y_pred_nb)
                st.success("Confusion Matrix:", confusion_matrix_nb)
                st.success("Classification Report:", classification_report_nb)
                st.success("Accuracy Score:", accuracy_score_nb)
            except ValueError:
                rmse_nb = np.sqrt(np.mean((y_test - y_pred_nb) ** 2))
                mae_nb = mean_absolute_error(y_test, y_pred_nb)
                st.success("Root Mean Squared Error:", rmse_nb)
                st.success("Mean Absolute Error:", mae_nb)

        if option == 'SVM' in selected_options:
            model_svc = SVC(gamma='scale')
            model_svc.fit(X_train, y_train)
            y_pred_svc = model_svc.predict(X_test)
            try:
                confusion_matrix_svc = confusion_matrix(y_test, y_pred_svc)
                classification_report_svc = classification_report(y_test, y_pred_svc)
                accuracy_score_svc = accuracy_score(y_test, y_pred_svc)
                st.success("Confusion Matrix:", confusion_matrix_svc)
                st.success("Classification Report:", classification_report_svc)
                st.success("Accuracy Score:", accuracy_score_svc)
            except ValueError:
                rmse_svc = np.sqrt(np.mean((y_test - y_pred_svc) ** 2))
                mae_svc = mean_absolute_error(y_test, y_pred_svc)
                st.success("Root Mean Squared Error:", rmse_svc)
                st.success("Mean Absolute Error:", mae_svc)

        if option == 'Decision tree' in selected_options:
            model_tree = DecisionTreeClassifier()
            model_tree.fit(X_train, y_train)
            y_pred_tree = model_tree.predict(X_test)
            try:
                confusion_matrix_tree = confusion_matrix(y_test, y_pred_tree)
                classification_report_tree = classification_report(y_test, y_pred_tree)
                accuracy_score_tree = accuracy_score(y_test, y_pred_tree)
                st.success("Confusion Matrix:", confusion_matrix_tree)
                st.success("Classification Report:", classification_report_tree)
                st.success("Accuracy Score:", accuracy_score_tree)
            except ValueError:
                rmse_tree = np.sqrt(np.mean((y_test - y_pred_tree) ** 2))
                mae_tree = mean_absolute_error(y_test, y_pred_tree)
                st.success("Root Mean Squared Error:", rmse_tree)
                st.success("Mean Absolute Error:", mae_tree)

        if option == 'Random forest' in selected_options:
            model_forest = RandomForestClassifier()
            model_forest.fit(X_train, y_train)
            y_pred_forest = model_forest.predict(X_test)
            try:
                confusion_matrix_forest = confusion_matrix(y_test, y_pred_forest)
                classification_report_forest = classification_report(y_test, y_pred_forest)
                accuracy_score_forest = accuracy_score(y_test, y_pred_forest)
                st.success("Confusion Matrix:", confusion_matrix_forest)
                st.success("Classification Report:", classification_report_forest)
                st.success("Accuracy Score:", accuracy_score_forest)
            except ValueError:
                rmse_forest = np.sqrt(np.mean((y_test - y_pred_forest) ** 2))
                mae_forest = mean_absolute_error(y_test, y_pred_forest)
                st.success("Root Mean Squared Error:", rmse_forest)
                st.success("Mean Absolute Error:", mae_forest)

        # Overview outcome
        st.write(pd.DataFrame({
            'Metrics': ['Confusion matrix', 'Classification report', 'Accuracy score', 'RMSE', 'MAE'],
            'Linear regression': [confusion_matrix_lin, classification_report_lin, accuracy_score_lin, rmse_lin,
                                  mae_lin],
            'Logistic regression': [confusion_matrix_log, classification_report_log, accuracy_score_log, rmse_log,
                                    mae_log],
            'KNN': [confusion_matrix_knn, classification_report_knn, accuracy_score_knn, rmse_knn, mae_knn],
            'Gaussian Naive Bayes': [confusion_matrix_nb, classification_report_nb, accuracy_score_nb, rmse_nb, mae_nb],
            'Support Vector Machines': [confusion_matrix_svc, classification_report_svc, accuracy_score_svc, rmse_svc,
                                        mae_svc],
            'Decision trees': [confusion_matrix_tree, classification_report_tree, accuracy_score_tree, rmse_tree,
                               mae_tree],
            'Random forest': [confusion_matrix_forest, classification_report_forest, accuracy_score_forest, rmse_forest,
                              mae_forest],
        }))


