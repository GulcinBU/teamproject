
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo



st.set_page_config(page_title="Hello", page_icon="👋")

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



#st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)




st.write("# Welcome to DataLimn!")



st.markdown(
    """
    DataLimn is your go-to open-source framework for machine learning and data science projects. The easy to use app gives greater insight and detailed information about your data. You simply upload a CSV file, follow some steps, and get your results!
### What does DataLimn do?
DataLimn is a tool designed to analyse patterns and make predictions without having to have previous knowledge of coding. This no-code machine learning app evaluates and interprets your data, transforming it into an understandable story that can be of great value in business decision-making processes. 
# Module description
### Data uploading and preparation
Upload a CSV file with your data. Once the data is uploaded the user can change the data type per column and select the training size. Furthermore , select the columns you would want to take into consideration as well as the column where you want to base your predictions on.
### Machine learning and data visualisation
The next step is to select one (or multiple) learning model(s) based on your data and the outcome you would like to see. Now it is time for our app to do the hard work. 
The app will give you predictions and probabilities based  on your data input, and some statistical graphs can be selected. Moreover, you can modify your input and generate a new outcome based on the features given. 

"""
)

st.write(' ')
st.write(' ')
next = st.button("next")
if next:
    switch_page("userlogin")
