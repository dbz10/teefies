from pickykitty import app
from flask import render_template, request
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

user = 'danielben-zion'
host = 'localhost'
dbname = 'catfood_db'
db = create_engine(f'postgres://{user}{host}/{dbname}')
con = None
con = psycopg2.connect(database = dbname, user = user)


from .model.predict import get_similar_items
from .model.misc import check_items_valid, filter_allergens



@app.route('/')
@app.route('/index')
def index():
	user = {'nickname' : 'Hooman'}
	return render_template("index.html",
		title = 'Home',
		user = user)

@app.route('/results', methods = ['GET','POST'])
def selection_results():

	products = request.args

	positive = [products.get(f'pos_{i}') for i in range(3) if products.get(f'pos_{i}')]
	negative = [products.get(f'neg_{i}') for i in range(3) if products.get(f'neg_{i}')]

	allergen_checkboxes = request.form.getlist('allergen_checkbox')


	liked = ', '.join(positive)
	disliked = ', '.join(negative)

	if not liked:
		liked = 'nothing'

	if not disliked:
		disliked = 'nothing'

	try:
		check_items_valid(products.values())
	except KeyError:
		return render_template("keyerr.html")

	if negative:
		allergens = filter_allergens(negative)
	else:
		allergens = []

	# adding support for checkbox remember state
	allergen_data = [(allergen,'checked' if allergen in allergen_checkboxes else []) for allergen in allergens]
	print(allergen_data)


	similar_items = get_similar_items(positive = positive, negative = negative)

	# let's pretend as if we're doing some SQL


	query = f""" SELECT product, price, num_cans, price_per_oz, url, ingredients
				  FROM product_info_table
				  WHERE product in {similar_items} """
	

	result_data = pd.read_sql_query(query,con)


	output = []
	recs_count = 0
	for item in similar_items:
		row = result_data.loc[result_data['product']==item]
		
		checkallergens = any([allergen in row['ingredients'].values[0].lower() for allergen in allergen_checkboxes])



		if checkallergens:
			continue
		else:
			# let just nicely format the price per oz
			price_per_oz = '$ %0.2f' % row['price_per_oz']


			output.append(dict(name=row['product'].values[0],
							   price=row['price'].values[0],
							   num_cans=row['num_cans'].values[0],
							   price_per_oz=price_per_oz,
							   url=row['url'].values[0]))

			recs_count += 1

		if recs_count > 4:
			break 




	testallergens = ['Chicken','Fish']
	return render_template("results.html", liked=liked, disliked = disliked, output = output, 
		allergen_data = allergen_data )



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
