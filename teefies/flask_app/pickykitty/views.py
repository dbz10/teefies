from pickykitty import app
from flask import render_template
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

user = 'danielben-zion'
host = 'localhost'
dbname = 'birth_db'
db = create_engine(f'postgres://{user}{host}/{dbname}')
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index():
	user = {'nickname' : 'dbz'}
	return render_template("index.html",
		title = 'Home',
		user = user)

@app.route('/db')
def birth_page():
	sql_query = """
				SELECT * 
				FROM birth_data_table
				WHERE delivery_method = 'Cesarean';
				"""

	query_results = pd.read_sql_query(sql_query, con)
	births = ""
	for i in range(10):
		births += query_results.iloc[i]["birth_month"]
		births += "<br>"
	return births
