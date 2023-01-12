
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
    DataLimn is an open-source app framework built specifically for
    Machine Learning and Data Science projects. You will get a detailed information 
    about your data when you simply upload your csv file and follow the instructions. 

    ### What is DataLimn?
    AppName is an AI based tool that can get your data set via uploading, 
    dissolve patterns in your data, can interpret the result, 
    and can then produce an output story that is understandable 
    to a business user based on the context.
     It is able to pro-actively analyse data on behalf of users and 
     generate smart feeds using natural language generation techniques 
     which can then be consumed easily by business users with very less efforts. 
     The application has been built keeping in mind a rather elementary 
     user and is hence, easily usable and understandable. 
     This also uses a multipage implementation of 
     Streamlit Library using Class based pages.
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### Features
    Given data/analytics output, the tool can 
    - turn the data into interactive data stories based on the given data
    - generate deep insights, infer pattern and help in business decisions.
    - provide personalization profiles; these could be represented as meta data describing what would be of interest to a given user.
    - generate reports understandable to a business user with interactive and intuitive interface.
    ### Module Description
    - Data Upload
    This module deals with the data upload. It can take csv and excel files. As soon as the data is uploaded, it creates a copy of the data to ensure that we don't have to read the data multiple times. It also saves the columns and their data types along with displaying them for the user. This is used to upload and save the data and it's column types which will be further needed at a later stage.
    - Data Preparation 
    Once the column types are saved in the metadata, we need to give the user the option to change the type. This is to ensure that the automatic column tagging can be overridden if the user wishes. For example a binary column with 0 and 1s can be tagged as numerical and the user might have to correct it. The three data types available are:
    - Data Visualization
    generate deep insights, infer pattern and help in business decisions.
    - Machine Learning
    provide personalization profiles; these could be represented as meta data describing what would be of interest to a given user.


"""
)


next = st.button("next")
if next:
    switch_page("userlogin")
