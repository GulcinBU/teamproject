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
st.write('Please upload your cleaned data which you downloaded in the "Upload Data" section.')

#upload cleaned file
uploaded_files = st.file_uploader("Choose  CSV file(s)", accept_multiple_files=True, type=['csv'])
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        columnheaders = df.columns.tolist()
        y = st.selectbox('Select y value', columnheaders)
        x = st.selectbox('Select x values', columnheaders)
        st.line_chart(df, x=x, y=y)
        st.area_chart(df, x=x, y=y)
        st.bar_chart(df, x=x, y=y)



 #   scatterplot= df.plot(x='x', y='y', kind='scatter')
   # lineplot = df.plot(x='x', y='y', kind='line')
 #   histplot= df.plot(x='x', y='y', kind='hist')
 #   pieplot= df.plot(x='x', y='y', kind='pie')
 #   cols_plots1 = st.columns(2)
 #   cols_plots1[0].plotly_chart(scatterplot, use_container_width=True)
  #  cols_plots1[1].plotly_chart(lineplot, use_container_width=True)
   # cols_plots2 = st.columns(2)
  #  cols_plots2[0].plotly_chart(histplot, use_container_width=True)
  #  cols_plots2[1].plotly_chart(pieplot, use_container_width=True)



st.write(' ')
st.write(' ')

next = st.button("next")
if next:
    switch_page("machinelearning")