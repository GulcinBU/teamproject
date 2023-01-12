import streamlit as st
import streamlit_option_menu
from streamlit_extras.switch_page_button import switch_page
import csv
import json
import pandas as pd
import pymongo
import certifi
import uuid
import io
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


client = pymongo.MongoClient("mongodb+srv://gulcin:Gevece98@c2codeequals.ilgclzn.mongodb.net/?retryWrites=true&w=majority" , tlsCAFile = certifi.where( ) )

db = client["UploadedFiles"]
col = db ["Filedetails"]
st.title('Upload Your File')
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
		col.insert_one(file_details)

		st.subheader('Details of Your File')
		st.write(file_details1)

		st.subheader('Your Dataframe')
		st.dataframe(df.head())

		st.subheader(' General Information of Your Data')
		buffer = io.StringIO()
		df.info(buf=buffer)
		s = buffer.getvalue()
		st.text(s)


		st.subheader(' Description of Your Data')
		dfd = df.describe(include='all')
		st.write(dfd)

		st.subheader('Dropping Duplicates')

		st.write(' Number of rows  is : {}'.format(df.shape[0]))
		# cleaning data
		# 1 drop duplicates
		df_1 = df.drop_duplicates(subset=None, keep="first", inplace=False)
		st.write('After dropping duplicates, number of rows  is:  {} '.format(df_1.shape[0]))

		st.subheader('Splitting Strings and Integers')
		st.write(" Your data may contain some columns which have both string and numbers. For example: 10 km ")
		st.write('I must remove strings in these columns to run numerical calculations. Please help me !')
		st.write(' Choose all columns name to fix them: ')

		columnheaders = df_1.columns.tolist()
		Header = st.multiselect('Select multiple columns to fix', columnheaders, default=None)
		for i in Header:
			column_i = df_1[i]
#			st.write(column_i)
			List_i = list(column_i)
			new_list_i = []
			for x in List_i:
				try:
					new_num = x.split(' ')[0]
					new_list_i.append(float(new_num))
				except:
					new_list_i.append(0)

#			st.write(new_list_i)
			series_i = pd.Series(new_list_i)
			df_1[i] = series_i.values
	st.write("Please note that you see default dataframe in the beginning. This dataframe will be restored as soon as your selection is done.")
	st.write(df_1)

	st.subheader('Handling Missing Values')
	for i in df_1:
		if df_1[i].dtypes != 'object' :
			median_i = df_1[i].median()
			df_1[i] = df_1[i].fillna(median_i)
			df_2 = df_1

	st.write(df_2)
	st.subheader(' General Information of Your Data After Missing Values Handling')
	buffer = io.StringIO()
	df_2.info(buf=buffer)
	s = buffer.getvalue()
	st.text(s)

	st.subheader(' Download Your Cleaned Data ')
	st.write( ' Data handling process is done! Please save the new clean data for the next steps!')


	def convert_df(df_2):
		return df_2.to_csv(index=False).encode('utf-8')


	csv = convert_df(df_2)

	st.download_button(
		"Press to Download",
		csv,
		"file.csv",
		"text/csv",
		key='download-csv'
	)




next = st.button("next")
if next:
    switch_page("visualizedata")





























