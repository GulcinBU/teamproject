import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


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

st.title('Get Plots of Your Data')
st.subheader('Please upload your cleaned data which you downloaded in the "Upload Data" section.')

#upload cleaned file
uploaded_files = st.file_uploader("Choose  CSV file(s)", accept_multiple_files=True, type=['csv'])
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        columnheaders = df.columns.tolist()
        st.subheader('Please select x and y variables for Scatter Plot, Line Plot, Bar Plot and Box Plot')
        feature_y = st.selectbox('Select y value', columnheaders)
        feature_x = st.selectbox('Select x values', columnheaders)
        st.subheader('Please select a single variable for Pie Plot')
        feature_z = st.selectbox('Select a value', columnheaders)

        fig1 = px.scatter(df, x=feature_x, y=feature_y)
        fig2 = px.line(df, x=feature_x, y=feature_y)
        fig3 =px.bar(df, x=feature_x, y=feature_y)
        fig4 = px.box(df, x=feature_x, y=feature_y)
        fig5 = px.pie(df, feature_z)
        st.subheader('Scatter Plot')
        st.plotly_chart(fig1)
        st.subheader('Line Plot')
        st.plotly_chart(fig2)
        st.subheader('Bar Plot')
        st.plotly_chart(fig3)
        st.subheader('Box Plot')
        st.plotly_chart(fig4)
        st.subheader('Pie Plot')
        st.plotly_chart(fig5)






st.write(' ')
st.write(' ')

next = st.button("next")
if next:
    switch_page("machinelearning")