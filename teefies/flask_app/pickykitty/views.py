from pickykitty import app
from flask import render_template, request
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

from .model.predict import get_similar_items
from .model.misc import check_items_valid

user = 'danielben-zion'
host = 'localhost'
dbname = 'catfood_db'
db = create_engine(f'postgres://{user}{host}/{dbname}')
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index():
	user = {'nickname' : 'Hooman'}
	return render_template("index.html",
		title = 'Home',
		user = user)

@app.route('/results')
def selection_results():
	products = request.args

	positive = [products.get(f'pos_{i}') for i in range(3) if products.get(f'pos_{i}')]
	negative = [products.get(f'neg_{i}') for i in range(3) if products.get(f'neg_{i}')]

	


	liked = ', '.join(positive)
	disliked = ', '.join(negative)

	if not liked:
		liked = 'nothing'

	if not disliked:
		disliked = 'nothing'

	check_items_valid(products.values())

	similar_items = get_similar_items(positive = positive, negative = negative)

	# let's pretend as if we're doing some SQL

	# query = f""" SELECT product AS name, price, oz_per_can AS size, price_per_oz, link AS url
	#			  FROM product_info_table
	#			  WHERE product in {similar_items} """
	#

	# result_data = pd.read_sql_query(query,con)


	return render_template("results.html", liked=liked, disliked = disliked, similar_items=similar_items)



# @app.route('/db')
# def birth_page():
# 	sql_query = """
# 				SELECT * 
# 				FROM birth_data_table
# 				WHERE delivery_method = 'Cesarean';
# 				"""

# 	query_results = pd.read_sql_query(sql_query, con)
# 	births = ""
# 	for i in range(10):
# 		births += query_results.iloc[i]["birth_month"]
# 		births += "<br>"
# 	return births


# @app.route('/db_fancy')
# def cesareans_page_fancy():
#     sql_query = """
#                SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
#                 """
#     query_results=pd.read_sql_query(sql_query,con)
#     births = []
#     for i in range(0,query_results.shape[0]):
#         births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#     return render_template('cesareans.html',births=births)

# @app.route('/input')
# def cesareans_input():
# 	return render_template("input.html")



# @app.route('/output')
# def cesareans_output():
#   #pull 'birth_month' from input field and store it
#   patient = request.args.get('birth_month')
#     #just select the Cesareans  from the birth dtabase for the month that the user inputs
#   query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
#   print(query)
#   query_results=pd.read_sql_query(query,con)
#   print(query_results)
#   births = []
#   for i in range(0,query_results.shape[0]):
#       births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
  
#   the_result = ModelIt(patient,births)
#   return render_template("output.html", births = births, the_result = the_result)
