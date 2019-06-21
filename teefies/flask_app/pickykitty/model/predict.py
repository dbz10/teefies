from gensim.models.doc2vec import Doc2Vec
import pickle
import pandas as pd

# /pickykitty/model/
label_decoder = pickle.load( 
	open('./pickykitty/model/label-decoder.pkl','rb') )

label_encoder = pickle.load( 
	open('./pickykitty/model/label-encoder.pkl','rb') )

model_dbow = Doc2Vec.load('./pickykitty/model/catfood-d2v-dbow.model')
model_dm = Doc2Vec.load('./pickykitty/model/catfood-d2v-dm.model')
# urls = pickle.load( open('./pickykitty/model/links.pkl','rb') )

def into_ranked_dataframe(similar_from_docvec):
	    """ Takes the output of doc2vec most_similar and puts it into
	    a dataframe thats nice to work with """
	    tmp = pd.DataFrame(similar_from_docvec,columns = ['product_label','sim_score'])
	    tmp['rank'] = tmp.index
	    tmp['name'] = tmp['product_label'].apply(lambda r: label_decoder[r])
	    
	    return tmp[['name','rank']].set_index('name')

def generate_rankings(positive=[],negative=[]):
	    num_returned = 100
	    
	    
	    similar_items_dbow = model_dbow.docvecs.most_similar(positive=positive,negative = negative,topn=num_returned)
	    similar_items_dm = model_dm.docvecs.most_similar(positive=positive, negative = negative,topn=num_returned)
	    
	    db_frame = into_ranked_dataframe(similar_items_dbow)
	    dm_frame = into_ranked_dataframe(similar_items_dm)

	    joined = db_frame.join(dm_frame,lsuffix='_db',rsuffix='_dm')
	    joined['avg_rank'] = joined.mean(axis=1)

	    joined.dropna(how='any',axis=0,inplace=True)

	    results = joined.sort_values(by='avg_rank')

	    
	    return results

def get_similar_items(positive = [], negative = [], num_results = 5):
	""" Returns most similar items computed from pretrained model """


	positives = [label_encoder[r] for r in positive]
	negatives = [label_encoder[r] for r in negative]

	# similar_items = model_dbow.docvecs.most_similar(positive = positives,
	# 										   negative = negatives,
	# 										   topn = num_results)


	similar_items_frame = generate_rankings(positive=positives,negative=negatives)

	decoded_items = [row[0] for row in similar_items_frame.head(5).iterrows()]


	return tuple(decoded_items)

	####################################################

	

	



if __name__ == '__main__':
	result = get_similar_items(positives = 'Earthborn Holistic Catalina Catch Grain-Free Natural Canned Cat & Kitten Food, 5.5-oz, case of 24')

	print(result)