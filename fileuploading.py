import streamlit as st
import streamlit_option_menu
import pymongo
import csv
import json
import pandas as pd
import pymongo
import certifi
import uuid
import pymongoimport
client = pymongo.MongoClient("mongodb+srv://gulcin:Gevece98@c2codeequals.ilgclzn.mongodb.net/?retryWrites=true&w=majority" , tlsCAFile = certifi.where( ) )

db = client["UploadedFiles"]
col = db ["Filedetails"]
filetags = st.text_input("Please enter tags of your data separating by comma")
uploaded_files = st.file_uploader("Choose  CSV file(s)", accept_multiple_files=True, type=['csv'])

for uploaded_file in uploaded_files:
	if uploaded_file is not None:
		df = pd.read_csv(uploaded_file)
		json_df = df.to_json()
		file_details1 = {"_id": uuid.uuid4().hex, "File Name": uploaded_file.name, "File Size": uploaded_file.size,
					"File Tags": filetags}
		file_details = {"_id": uuid.uuid4().hex, "File Name": uploaded_file.name, "File Size": uploaded_file.size,
						"File Tags": filetags, "Data": json_df}
		st.write(file_details1)
		st.dataframe(df.head())
		col.insert_one(file_details)

























