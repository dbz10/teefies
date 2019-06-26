import pickle
from gensim.models.doc2vec import Doc2Vec
from pickykitty.views import con
import pandas as pd

# /pickykitty/model/
label_decoder = pickle.load( 
	open('./pickykitty/model/label-decoder.pkl','rb') )

label_encoder = pickle.load( 
	open('./pickykitty/model/label-encoder.pkl','rb') )

# model = Doc2Vec.load('./pickykitty/model/catfood-d2v-dbow.model')


def check_items_valid(input):
	[label_encoder[item] for item in input if item]
	pass



def filter_allergens(input):

	common_allergens = ['chicken','beef','pork','turkey','corn','seafood',
                    'wheat gluten','soy','dairy','by-products']

	allergens_counter = pd.DataFrame.from_dict({item: 0 for item in common_allergens},
												orient='index',columns=['count'])


	if len(input) > 1:
		items = tuple(input)
		query = f""" SELECT product, ingredients
					FROM product_info_table
					WHERE product in {items}
		"""
	else:
		query = f""" SELECT product, ingredients
					FROM product_info_table
					WHERE product = '{input[0]}'
		"""

	query_results = pd.read_sql_query(query,con)

	for index,row in query_results.iterrows():
		for allergen in common_allergens:
			allergens_counter.loc[allergen] += (allergen in row['ingredients'].lower())

	
	potential_positives = allergens_counter.loc[allergens_counter['count']>=2].index.values


	return potential_positives
