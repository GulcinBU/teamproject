import streamlit as st
from streamlit_option_menu import option_menu
import login
with st.sidebar:
    option = option_menu("User login",["Register","Login","Update password"],orientation="vertical")

if option == "Register":
    with st.expander("Expand to register",expanded=True):
        cols = st.columns([1, 3, 1])
        name = cols[1].text_input("Enter the name")
        email = cols[1].text_input("Enter the email")
        password = cols[1].text_input("Enter the password",type="password")
        if st.button("Register"):
            result = login.userregistration(name,email,password)
            st.success(result)

if option == "Login":
    with st.expander("Expand to login",expanded=True):
        cols = st.columns([1, 3, 1])
        email = cols[1].text_input("Enter the email")
        password = cols[1].text_input("Enter the password",type="password")
        if cols[1].checkbox('login'):
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
