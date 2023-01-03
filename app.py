import streamlit as st
from streamlit_option_menu import option_menu
import login

with st.sidebar:
    option = option_menu("User login",["Register","Login","Update password"],orientation="vertical")

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
        cols = st.columns([1, 3, 1])
        email = cols[1].text_input("Enter your email address")
        if cols[1].button('Send link to my email'):
            result_request = login.sendtoken(email)
            st.success(result_request)

if option == "Reset password":
    with st.expander("Expand to reset password",expanded=True):
        cols = st.columns([1, 3, 1])
        password = cols[1].text_input("Enter new password", type="password")
        result_reset = login.resetpassword(password)
        st.success(result_reset)
        st.error(result_reset)