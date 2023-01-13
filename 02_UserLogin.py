
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
import login

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



option = option_menu("User Login",["Register","Login","Update password"],orientation="horizontal")

if option == "Register":
    with st.expander("Expand to register",expanded=True):
        cols = st.columns([1, 3, 1])
        name = cols[1].text_input("Enter your name")
        email = cols[1].text_input("Enter your email address")
        password = cols[1].text_input("Enter a password",type="password")
        if cols[1].button("Register"):
            result = login.userregistration(name,email,password)
            st.success(result)

if option == "Login":
    with st.expander("Expand to login",expanded=True):
        cols = st.columns([1, 3, 1])
        email = cols[1].text_input("Enter your email address")
        password = cols[1].text_input("Enter password",type="password")
        if cols[1].button('Login'):
            global result_login
            result_login = login.login(email,password)
            st.success(result_login)

if option == "Update password":
    with st.expander("Expand to update password",expanded=True):
        cols = st.columns([1,3,1])
        name = cols[1].text_input("Enter name")
        email = cols[1].text_input("Enter email")
        password = cols[1].text_input("Enter new password", type="password")
        if cols[1].checkbox("Reset"):
            result_password = login.updatepassword(name,email,password)
            st.success(result_password)

st.write(' ')
st.write(' ')
next = st.button("next")
if next:
    switch_page("uploaddata")