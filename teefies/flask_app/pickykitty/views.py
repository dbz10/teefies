from pickykitty import app
from flask import render_template, request
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

from config import Values

user = Values.user
dbname = Values.dbname
host = Values.host
password = Values.password 


engine = create_engine('postgres://%s:%s@localhost/%s'%(user,password,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user,password=password, host=host)


from .model.predict import get_similar_items
from .model.misc import check_items_valid, filter_allergens

def format_lists_for_printing(list):

	if not list:
		out = []
	else:
		out = list

	return out



@app.route('/')
@app.route('/index')
def index():
	user = {'nickname' : 'Cat Parent'}
	return render_template("index.html",
		title = 'Home',
		user = user)

@app.route('/results', methods = ['GET','POST'])
def selection_results():


	products = request.args
	
	if len([item for item in products.values() if item]) == 0:
		query = """
		SELECT product_info_table.product, price, price_per_oz, avg_rating, url
		FROM product_info_table
		INNER JOIN
		(
		SELECT product, AVG(rating) as avg_rating
		FROM reviews_table
		GROUP BY product
		) AS mean_rating
		ON product_info_table.product = mean_rating.product
		ORDER BY avg_rating DESC
		LIMIT 5
		"""

		result_data = pd.read_sql_query(query,con)


		output = []
		for index,row in result_data.iterrows():
			print(row)
			# let just nicely format the price per oz
			price_per_oz = '$ %0.2f' % row['price_per_oz']


			output.append(dict(name=row['product'],
							   price=row['price'],
							   price_per_oz=price_per_oz,
							   url=row['url']))


		return render_template("noinput.html",  output = output )






	

	positive = [products.get(f'pos_{i}') for i in range(3) if products.get(f'pos_{i}')]
	negative = [products.get(f'neg_{i}') for i in range(3) if products.get(f'neg_{i}')]

	duplicates = set(positive).intersection(set(negative))
	

	allergen_checkboxes = request.form.getlist('allergen_checkbox')


	liked = format_lists_for_printing(positive.copy())
	disliked = format_lists_for_printing(negative.copy())


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
							   price_per_oz=price_per_oz,
							   url=row['url'].values[0]))

			recs_count += 1

		if recs_count > 4:
			break 


	if duplicates:
		return render_template("results_warning.html", liked=liked, disliked = disliked, output = output, 
		allergen_data = allergen_data , duplicates=duplicates)

	testallergens = ['Chicken','Fish']
	return render_template("results.html", liked=liked, disliked = disliked, output = output, 
		allergen_data = allergen_data )
