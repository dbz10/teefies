# Imports
# from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
# import csv
import multiprocessing
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
import numpy as np
import itertools
import json

# Parameters for "grid search"
vector_sizes = [100,150,200,250,300]
alphas = [0.025, 0.05, 0.075]
windows = [4,7,10]
epochs = [20,30,40]

parms = [dict(vector_size=vector_size,alpha=alpha,window=window,epochs=epochs) 
for (vector_size,alpha,window,epochs) in itertools.product(vector_sizes,alphas,windows,epochs)]

with open( 'parms.txt','w') as pl:
	for (i,parm) in enumerate(parms):
		pl.write(f'{i}: ')
		pl.write(json.dumps(parm))
		pl.write("\n")




# Load in the data and display some basic info
product_info = pd.read_csv('../data/CatfoodProductInfo.csv')
reviews = pd.read_csv('../data/CatfoodReviewsInfo.csv')
df = reviews.join(product_info.set_index('product'), on='product',how='left')
df = df.dropna(axis=0,how='any')

ninit = len(set(df['product']))
print(f'Prior to filtering out products with less than 50 reviews, there are {ninit} products')

filter = df.groupby('product')['rating'].count() >= 50
df = df[filter[df['product']].values]

nfin = len(set(df['product']))
print(f'After filtering out products with less than 50 reviews, there are {nfin} products')

prod_info_filter = [product in set(df['product']) for product in product_info['product']]
product_info = product_info[prod_info_filter]

# Drop variety packs that somehow made it through
print(f'Before removing straggler variety packs: {len(df)}')
exclude_words = ['Variety',]
for word in exclude_words:
    df = df[~df['product'].str.contains(word)]
    product_info = product_info[~product_info['product'].str.contains(word)]
    
print(f'After removing straggler variety packs: {len(df)}')

brandnames = set(df['brand'].unique())
print(f'There are {len(brandnames)} brands represented across our reviews.')

nprods = len(df.groupby('product'))
nrevs = len(df)
print(f'After all of that, there are {nrevs} reviews across {nprods} products')


# standardize text
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    for brandname in brandnames:
        df[text_field] = df[text_field].str.replace(brandname.lower(),"")

        
    return df

# Label encode the names
le = preprocessing.LabelEncoder()
df['product_label']=le.fit_transform(df['product'])


df_clean = standardize_text(df,'review_text')

# Tokenize and remove stop words.

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
df_clean["tokens"] = df_clean["review_text"].apply(tokenizer.tokenize)

df_clean['tokens'] = df_clean['tokens'].apply(lambda x: ' '.join( [item for item in x if item not in stop_words]))
# df_clean['tokens'] = df_clean['tokens'].apply(lambda x: [item for item in x if item not in stop_words])


all_tokens = [t.split() for t in df_clean['tokens']]
phrases = Phrases(all_tokens)
bigram = Phraser(phrases)
trigram_phrases = Phrases(bigram[all_tokens])
trigram = Phraser(phrases)

df_clean['input'] = df_clean['tokens'].apply(lambda x: bigram[x.split(' ')])
# df_clean['input'] = df_clean['tokens'].apply(lambda x: x.split(' '))



# train, test = train_test_split(df[['review_text','product_label']],test_size=0.0)

data = df_clean[['input','product_label']]



data['n_words'] = data['input'].apply(lambda r: len(r))
print(f'Number of reviews prior to dropping short ones {len(data)}')
data = data.loc[data['n_words']>=10]
print(f'Number of reviews after dropping short ones {len(data)}')
# data.head(5)

data_tagged = data.apply(
    lambda r: TaggedDocument(words=r['input'], tags=[r.product_label]), axis=1)

cores = multiprocessing.cpu_count()

label_decoder = df[['product_label','product']].set_index('product_label').to_dict()['product']
label_encoder = df[['product_label','product']].set_index('product').to_dict()['product_label']




for (trialno,parm) in tqdm(enumerate(parms)):

	vector_size = parm['vector_size']
	alpha = parm['alpha']
	window = parm['window']
	epochs = parm['epochs']

	model_dbow = Doc2Vec(dm=0, vector_size=vector_size, window=window, negative=5, hs=0, min_count=2,
	 					sample = 0, workers=cores, alpha=alpha, min_alpha=0.001)
	model_dm = Doc2Vec(dm=1, vector_size=vector_size, window=window, negative=5, hs=0, min_count=2, sample=0,
	                   workers=cores, alpha=alpha, min_alpha = 0.001)


	model_dbow.build_vocab(data_tagged.values)
	model_dm.build_vocab(data_tagged.values)

	train_data = utils.shuffle(data_tagged)

	model_dbow.train(train_data, total_examples=len(train_data), epochs=epochs)
	model_dm.train(train_data, total_examples=len(train_data), epochs=epochs)

	author_count = df.groupby('review_author')['review_author'].count()
	authorgroup = author_count[(author_count > 5) & (author_count < 15)]


	def scale_scores(df,field):
	    scaleby = max(np.abs(df[field].min()),np.abs(df[field].max()))
	    df[field] = df[field]/scaleby
	    return df

	def generate_val_data(user):
	    num_returned = 1000
	    userdata = df[df['review_author']==user]
	    
	    if len(userdata.groupby('rating').count()) == 1:
	        return
	    
	    

	    
	    low_rankings = userdata[userdata['rating'] <= 3].sort_values(by='rating',ascending=True)
	    high_rankings = userdata[userdata['rating'] >= 4].sort_values(by='rating',ascending=False)
	    
	    
	    # if len(low_rankings) < 3:
	    #     return
	    
	    
	    negatives = [val for val in low_rankings.head(2)['product_label']]
	    positives = [val for val in high_rankings.head(2)['product_label']]
	    
	    similar_items_dbow = model_dbow.docvecs.most_similar(positive=positives,negative = negatives,topn=num_returned)
	    similar_items_dm = model_dm.docvecs.most_similar(positive=positives,negative = negatives,topn=num_returned)

	    decoded_dbow = [(label_decoder[label],similarity) for (label,similarity) in similar_items_dbow]
	    decoded_dm = [(label_decoder[label],similarity) for (label,similarity) in similar_items_dm]

	    dbow_results = pd.DataFrame(decoded_dbow,columns=['product','sim_score'])
	    dbow_results = scale_scores(dbow_results,'sim_score')

	    dm_results = pd.DataFrame(decoded_dm,columns=['product','sim_score'])
	    dm_results = scale_scores(dm_results,'sim_score')

	    combined_results = dbow_results.set_index('product').join(dm_results.set_index('product'), how = 'left', 
	                                        lsuffix = '_db', rsuffix = '_dm')


	    combined_results['avg_sim'] = (combined_results['sim_score_db'] + combined_results['sim_score_dm'])/2
	    
	    tmp = userdata[['product','rating']].set_index('product')
	    val = tmp.join(combined_results,how='left')
	    val.dropna(how='any',axis=0,inplace=True)
	    
	#     val['rating'] = val['rating'] - val['rating'].mean()
	    
	    return val

	val = pd.concat([generate_val_data(user) for user in authorgroup.index],axis=0)

	fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(12,4))


	a = sns.boxplot(x='sim_score_dm',y='rating',
	                    dodge=False,linewidth=2.5,orient='h',data=val,ax = axes[0])
	a.invert_yaxis()

	b = sns.boxplot(x='sim_score_db',y='rating',
	                    dodge=False,linewidth=2.5,orient='h',data=val,ax = axes[1])
	b.invert_yaxis()

	c = sns.boxplot(x='avg_sim',y='rating',
	                    dodge=False,linewidth=2.5,orient='h',data=val,ax = axes[2])
	c.invert_yaxis()

	plt.savefig(f'plots/round1/sim-box-bigrams-{trialno}.png')

	plt.close('all')

print('Complete.')
