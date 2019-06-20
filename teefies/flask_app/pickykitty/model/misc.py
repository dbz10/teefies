import pickle
from gensim.models.doc2vec import Doc2Vec

# /pickykitty/model/
label_decoder = pickle.load( 
	open('./pickykitty/model/label-decoder.pkl','rb') )

label_encoder = pickle.load( 
	open('./pickykitty/model/label-encoder.pkl','rb') )

model = Doc2Vec.load('./pickykitty/model/catfood-d2v-dbow.model')


def check_items_valid(input):
	[label_encoder[item] for item in input if item]
	pass