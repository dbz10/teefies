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

def get_similar_items(positive = [], negative = [], num_results = 5):
	""" Returns most similar items computed from pretrained model """

	try:
		positive = [label_encoder[r] for r in positive]
		negative = [label_encoder[r] for r in negative]
	except KeyError:
		print('One of your inputs was not recognized')


	similar_items = model.docvecs.most_similar(positive = positive,
											   negative = negative,
											   topn = num_results)

	decoded_items = [label_decoder[label] for (label,similarity) in similar_items]

	columns = ['product','link']

	result = [ pd.DataFrame(data = {'product': [item] , 'link': [urls[item]]}) for item in decoded_items]

	return result



if __name__ == '__main__':
	result = get_similar_items(positives = 'Earthborn Holistic Catalina Catch Grain-Free Natural Canned Cat & Kitten Food, 5.5-oz, case of 24')

	print(result)