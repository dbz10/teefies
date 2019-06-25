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
from scipy import interp
import itertools
import json
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, roc_auc_score, accuracy_score


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

def make_val_boxplots(val):
    
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

def make_val_boxplot_avg_sim_only(val):
	fig = plt.figure(figsize=(8,6))
	plt.rcParams.update({'font.size': 16})

	a = sns.boxplot(x='avg_sim',y='rating',
		dodge=False,linewidth=2.5,orient='h',data=val)

	plt.xlabel('Average Similarity')
	plt.ylabel('Individual User Rating on Product')
	plt.title('Similarity Based Recommendations')

	a.invert_yaxis()

def get_threshold(val,feature_field):
	""" Gets the threshold value of similarity for which
	the probability for class = good is >= 75% """
	lr = LogisticRegression(random_state=40,class_weight='balanced')
	X = lrdf[feature_field].values
	y = lrdf["class"].values

	lr.fit(X.reshape(-1,1),y)

	sim_vals = np.arange(-1,1,0.01)
	probs = lr.predict_proba(sim_vals.reshape(-1,1))[:,0]
	threshold = sim_vals[np.argwhere(probs>0.25).max()]

	return threshold

class lr_wrapper(object):
	""" A wrapper for LinearRegression() that packages 
	several commonly-performed tasks """

	def __init__(self,df,test_split=0.5,feature_columns=[],y_column=[]):
		self.clf = LogisticRegression(solver='lbfgs',class_weight='balanced')
		self.df = df
		self.feature_columns = feature_columns
		self.y_column = y_column
		self.test_split = test_split

	def fit_and_return_probas(self):
		""" Performs train_test_split, fits linear regression, predicts probabilities,
		and returns y_test, y_probas"""
		X = self.df[self.feature_columns]
		y = self.df[self.y_column]
		X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = self.test_split,stratify = y)

		self.clf.fit(X_train,y_train)
		y_probas = self.clf.predict_proba(X_test)[:,1]

		return y_test,y_probas

	def fit_and_return_preds(self):
		""" Performs train_test_split, fits linear regression, predicts,
		and returns y_test, y_preds"""
		X = self.df[self.feature_columns]
		y = self.df[self.y_column]
		X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = self.test_split,stratify = y)

		self.clf.fit(X_train,y_train)
		y_preds = self.clf.predict(X_test)

		return y_test,y_preds




def get_rocauc(val,num_iterations):
	""" Trains a logistic regression and calculates the roc auc 
	for classifying products as >=4 stars """

	recalls = np.zeros(num_iterations)
	precisions = np.zeros(num_iterations)
	f1s = np.zeros(num_iterations)
	roc_aucs = np.zeros(num_iterations)


	factory = lr_wrapper(val,feature_columns=['sim_score_db','sim_score_dm'],y_column='class')

	for z in range(num_iterations):
		# Slightly annoying thing here that each call to factory uses its own 
		# train_test_split, so y_test used for recalls will be different than
		# y_test used in roc aucs

		y_test,y_preds = factory.fit_and_return_preds()
		recalls[z] = recall_score(y_test,y_preds)
		precisions[z] = precision_score(y_test,y_preds)
		f1s[z] = f1_score(y_test,y_preds)

		y_test,y_probas = factory.fit_and_return_probas()
		roc_aucs[z] = roc_auc_score(y_test, y_probas)


	# print(roc_aucs)
	return np.mean(recalls),np.mean(precisions),np.mean(f1s),np.mean(roc_aucs)

def make_roc_curve_data(val,num_iterations):
	""" Makes the data to be used by make_roc_curve_confidence"""
	roc_aucs = np.zeros(num_iterations)
	tprs = []
	base_fpr = np.linspace(0, 1, 101)


	factory = lr_wrapper(val,feature_columns=['sim_score_db','sim_score_dm'],y_column='class')

	for z in range(num_iterations):
		y_test, y_probas = factory.fit_and_return_probas()
		roc_aucs[z] = roc_auc_score(y_test, y_probas)

		fpr, tpr, _ = roc_curve(y_test, y_probas)

		tpr = interp(base_fpr, fpr, tpr)
		tpr[0] = 0.0
		tprs.append(tpr)

	# print(roc_aucs)
	return np.mean(roc_aucs), tprs

def make_roc_curve_confidence(val,num_iterations):
	# from https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates/187003

	roc_auc, tprs = make_roc_curve_data(val,num_iterations)

	tprs = np.array(tprs)
	base_fpr = np.linspace(0, 1, 101)
	mean_tprs = tprs.mean(axis=0)
	std = tprs.std(axis=0)

	tprs_upper = np.minimum(mean_tprs + std, 1)
	tprs_lower = mean_tprs - std

	make_baseline_curve(validation_baseline,num_iterations)
	plt.plot(base_fpr, mean_tprs, 'b',label='My AUROC: %.3f' % roc_auc)
	plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	# plt.axes().set_aspect('equal', 'datalim')
	plt.legend()
	plt.show()

def make_baseline_curve(df,num_iterations):
	factory = lr_wrapper(df,feature_columns=['rating_mean'],y_column='class')

	roc_aucs = np.zeros(num_iterations)
	tprs = []
	base_fpr = np.linspace(0, 1, 101)

	for z in range(num_iterations):
		y_test, y_probas = factory.fit_and_return_probas()
		roc_aucs[z] = roc_auc_score(y_test, y_probas)

		fpr, tpr, _ = roc_curve(y_test, y_probas)

		tpr = interp(base_fpr, fpr, tpr)
		tpr[0] = 0.0
		tprs.append(tpr)


	tprs = np.array(tprs)
	mean_tprs = tprs.mean(axis=0)
	roc_auc = roc_aucs.mean()

	plt.plot(base_fpr,mean_tprs,'g',label='Baseline AUROC: %.2f' % roc_auc)




try: 
	data = pd.read_csv('prepared_data.csv')
except:
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

	data = df_clean.copy()



	data['n_words'] = data['input'].apply(lambda r: len(r))
	print(f'Number of reviews prior to dropping short ones {len(data)}')
	data = data.loc[data['n_words']>=10]
	print(f'Number of reviews after dropping short ones {len(data)}')
	# data.head(5)

	data.to_csv('prepared_data.csv')


data_tagged = data.apply(
    lambda r: TaggedDocument(words=r['input'], tags=[r.product_label]), axis=1)

cores = multiprocessing.cpu_count()

label_decoder = data[['product_label','product']].set_index('product_label').to_dict()['product']
label_encoder = data[['product_label','product']].set_index('product').to_dict()['product_label']

author_count = data.groupby('review_author')['review_author'].count()
authorgroup = author_count[(author_count > 5) & (author_count < 20)]

mean_product_ratings = data[['product','rating']].groupby('product').mean()

def generate_baseline_for_validation(user):
	num_returned = 1000
	userdata = data[data['review_author']==user]

	tmp = userdata[['product','rating']].set_index('product')

	val = tmp.join(mean_product_ratings,how='left',rsuffix='_mean')

	return val

validation_baseline = pd.concat([generate_baseline_for_validation(user) for user in authorgroup.index],axis=0)
validation_baseline['class'] = validation_baseline['rating'].apply(lambda x: 0 if x > 3 else 1)



columns = ['param set', 'threshold_dbow','threshold_dm','threshold_avg']
with open('validation_metrics.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)

AUC_RESULTS = np.zeros((len(parms),len(parms)))
PRECISION_RESULTS = np.zeros((len(parms),len(parms)))
F1_RESULTS = np.zeros((len(parms),len(parms)))
RECALL_RESULTS = np.zeros((len(parms),len(parms)))

for p1,p2 in tqdm(itertools.product(enumerate(parms),enumerate(parms))):


	parms_no_dbow = p1[0]
	parms_dbow = p1[1]
	parms_no_dm = p2[0]
	parms_dm = p2[1]

	# if (parms_no_dbow,parms_no_dm) != (0,0):
	# 	continue


	try:
		model_dbow = Doc2Vec.load(f'saved_models/catfood-d2v-dbow-{parms_no_dbow}.model')
		model_dm = Doc2Vec.load(f'saved_models/catfood-d2v-dm-{parms_no_dm}.model')
		# print(f'Successfuly loaded models dbow: {parms_no_dbow}, dm: {parms_no_dm}')
	except:
		model_dbow = Doc2Vec(dm=0, negative=5, hs=0, min_count=2,
		 					sample = 0,  min_alpha=0.001, vector_size = parms_dbow['vector_size'],
		 					alpha = parms_dbow['alpha'], window = parms_dbow['window'])
		model_dm = Doc2Vec(dm=1,  negative=5, hs=0, min_count=2, 
			sample=0,  min_alpha = 0.001, vector_size = parms_dm['vector_size'],
		 					alpha = parms_dm['alpha'], window = parms_dm['window'])


		model_dbow.build_vocab(data_tagged.values)
		model_dm.build_vocab(data_tagged.values)

		train_data = utils.shuffle(data_tagged)

		model_dbow.train(train_data, total_examples=len(train_data), epochs=parms_dbow['epochs'])
		model_dm.train(train_data, total_examples=len(train_data), epochs=parms_db['epochs'])

		model_dbow.save(f'saved_models/catfood-d2v-dbow-{parms_no_dbow}.model')
		model_dm.save(f'saved_models/catfood-d2v-dm-{parms_no_dm}.model')

	


	def scale_scores(df,field):
	    scaleby = max(np.abs(df[field].min()),np.abs(df[field].max()))
	    df[field] = df[field]/scaleby
	    return df

	def generate_val_data(user):
	    num_returned = 1000
	    userdata = data[data['review_author']==user]
	    
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
	    
	    
	    return val

	val = pd.concat([generate_val_data(user) for user in authorgroup.index],axis=0)

	# make_val_boxplots(val)
	# plt.savefig(f'plots/round1/sim-box-bigrams-dropped_short_reviews-{trialno}.png')
	# plt.close('all')

	# make_val_boxplot_avg_sim_only(val)
	# plt.savefig('plots/VAL_FOR_PRESENTATION.png',bbox_inches='tight')



	val['class'] = val['rating'].apply(lambda x: 0 if x > 3 else 1)

	recall,precision,f1,roc_auc = get_rocauc(val,50)

	# make_roc_curve_confidence(val,500)

	

	# ,fpr,tpr,thresholds

	AUC_RESULTS[parms_no_dbow,parms_no_dm] = roc_auc
	RECALL_RESULTS[parms_no_dbow,parms_no_dm] = recall
	PRECISION_RESULTS[parms_no_dbow,parms_no_dm] = precision
	F1_RESULTS[parms_no_dbow,parms_no_dm] = f1

	# print(f'Recall: {recall}, Precision: {precision}, F1: {f1}, ROC AUC: {roc_auc}')

	# Plotting ROC curve
	# plt.figure(figsize=(8, 8))
	# plt.plot(fpr, tpr, lw=2)
	# plt.plot([0, 1], [0, 1], 'k--')
	# plt.title('ROC (AUC=%0.3f)' % roc_auc)
	# plt.xlabel('FPR')
	# plt.ylabel('TPR')
	# plt.show()


	# val_metrics = [get_threshold(val,field) for field in ['sim_score_db','sim_score_dm','avg_sim']]
	# val_metrics.insert(0,trialno)

	# with open('validation_metrics.csv','a+') as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	writer.writerow(val_metrics)
	np.save('validation_results/auc_results.npy',AUC_RESULTS)
	np.save('validation_results/f1_results.npy',F1_RESULTS)
	np.save('validation_results/recall_results.npy',RECALL_RESULTS)
	np.save('validation_results/precision_results.npy',PRECISION_RESULTS)



# def find_max(array):
# 	result = np.where(array == np.amax(array))
# 	listOfCordinates = list(zip(result[0], result[1]))
# 	return listOfCordinates

# arrlist = [AUC_RESULTS,F1_RESULTS,RECALL_RESULTS,PRECISION_RESULTS]
# for name,result_array in zip(['auc','f1','recall','precision'],arrlist ):
# 	max_inds = find_max(result_array)
# 	print(f'Max value of {name} are at indices {max_inds}')
	


print('Complete.')
