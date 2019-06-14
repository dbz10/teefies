from gensim.models.doc2vec import Doc2Vec
import pickle
import pandas as pd

# /pickykitty/model/
label_decoder = pickle.load( 
	open('./pickykitty/model/label-decoder.pkl','rb') )

label_encoder = pickle.load( 
	open('./pickykitty/model/label-encoder.pkl','rb') )

model = Doc2Vec.load('./pickykitty/model/catfood-d2v.model')
urls = pickle.load( open('./pickykitty/model/links.pkl','rb') )

def get_similar_items(positives = [], negatives = [], num_results = 5):
	""" Returns most similar items computed from pretrained model """

	# print(positives[0])
	# pos_in = [label_encoder[x] for x in positives]
	# neg_in = [label_encoder[x] for x in negatives]

	product_in = str(positives)
	pos_in = label_encoder[product_in]
	print(pos_in)


	similar_items = model.docvecs.most_similar(positive = pos_in,
											   negative = [],
											   topn = num_results)

	# print(similar_items)
	# print( [label for (label,similarity) in similar_items])

	# print( label_encoder )
	# print( label_decoder )
	decoded_items = [label_decoder[label] for (label,similarity) in similar_items]

	columns = ['product','link']

	result = [ pd.DataFrame(data = {'product': [item] , 'link': [urls[item]]}) for item in decoded_items]

	return result



if __name__ == '__main__':
	result = get_similar_items(positives = 'Earthborn Holistic Catalina Catch Grain-Free Natural Canned Cat & Kitten Food, 5.5-oz, case of 24')

	print(result)