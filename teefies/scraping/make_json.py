import json
import pickle

all_links = pickle.load( open('links.pkl','rb') )
print(f'There are {len(all_links)} links.')

counter = 0
# for key in all_links.keys():
# 	name = ' '.join(key.split(',')[:-2])
# 	print(name)
# 	counter +=1
# 	if counter > 200:
# 		break

jl = json.dumps([dict(name=' '.join(key.split(',')[:-2])) for key in all_links.keys()])

# print(jl)

with open('foods.json','w') as json_file:
	json.dump(jl,json_file)


jl_loaded = json.load( open('foods.json','r') )
print(jl_loaded)