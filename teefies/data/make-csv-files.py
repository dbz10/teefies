import glob
import pickle
import re
import csv
import os
from bs4 import BeautifulSoup
import string
import json

basedir = '/Users/danielben-zion/Dropbox/insight/teefies/data'
num_items = len(glob.glob(f'{basedir}/html_pages/individual-items/*'))

all_links = pickle.load( open('links.pkl','rb') )
print(f'There are {len(all_links)} total links in all_links pickle.')
print(f'In the directory, there are {num_items} folders.')

def make_reviews_csv():
    # Reviews CSV file
    columns = ['product', 'review_author','rating','review_text','helpful_rank']
    with open('/Users/danielben-zion/Dropbox/insight/teefies/data/CatfoodReviewsInfo.csv','w') as catfoodcsv:
        writer = csv.writer(catfoodcsv)
        writer.writerow(columns)


        review_counter = 0
        product_counter = 0 
        for (product_name, product_link) in all_links.items():
            

            name=' '.join(product_name.split(',')[:-2])
            
            product_dir = product_name.translate(str.maketrans('', '', string.punctuation))
            product_dir = product_dir.replace(' ','-').lower()

            try:

                # print(product_dir)
                for reviewpage in glob.glob(f'{basedir}/html_pages/individual-items/{product_dir}/page*.html'):

                    # print(reviewpage)

                    review_soup = BeautifulSoup(open(reviewpage,'r'),'html')

                    for review in review_soup.find_all("li", {'itemprop': 'review'}):
                        rating = review.find('meta',{'itemprop':'ratingValue'})['content']
                        review_text = review.find('span',{'class' : 'ugc-list__review__display'}).text
                        review_author = review.find('span',{'itemprop' : 'author'}).text
                        helpful = review.find('button',{'class' : 'cw-btn cw-btn--white cw-btn--sm ugc__like js-like'}).span.text
                        writer.writerow([name,review_author,rating,review_text,helpful])
                        review_counter += 1

                product_counter += 1
            except: 
                continue

        print(f'Succesfully made csv file for {review_counter} reviews across {product_counter} products')



def make_product_csv():
    name_list_to_json = []
    # Product Info CSV file
    columns = ['product','brand', 'price','oz_per_can','num_cans','price_per_oz','ingredients']
    with open('/Users/danielben-zion/Dropbox/insight/teefies/data/CatfoodProductInfo.csv','w') as catfoodcsv:
        writer = csv.writer(catfoodcsv)
        writer.writerow(columns)


        counter = 0 
        for (product_name, product_link) in all_links.items():

            name=' '.join(product_name.split(',')[:-2])
            
            product_dir = product_name.translate(str.maketrans('', '', string.punctuation))
            product_dir = product_dir.replace(' ','-').lower()

            try:
                main_page_soup = BeautifulSoup(open(f'{basedir}/html_pages/individual-items/{product_dir}/main_page.html','r'),features="lxml")
            except:
                continue

            try: # just make sure we actually have html data for this product
                price = main_page_soup.find('span',{'class' : 'ga-eec__price'}).text.strip()
            except:
                continue 



        

            
            try:
                oz_per_can = float(re.search('(\d.?\d?)-oz',product_name).group(1))
                num_cans = float(re.search('case of (\d+)',product_name).group(1))
                ppo = float(price[1:])/(oz_per_can*num_cans)
            except: 
                continue
            

            name_list_to_json.append(dict(name=name))

            brand = main_page_soup.find('span',{'itemprop' : 'brand'}).text

            ingredients_block = main_page_soup.find('span', {'class': "cw-type__h2 Ingredients-title"})
            ingredients = ingredients_block.next_element.next_element.next_element.text.strip().split(',')
            

    #         print(f'Full product name: {product_name}')

    #         print(f"Name: {nn}" )
    #         print(f'oz_per_can: {opc}')
    #         print(f'num_cans: {ncans}')
    #         print(f'price: {price}')
            

            row = [name,brand,price,oz_per_can,num_cans,ppo,ingredients]
            writer.writerow(row)
            counter += 1

        with open('foods.json','w') as json_file:
            json.dump(name_list_to_json,json_file)

        print(f'Succesfully made csv data file for {counter} products')


if __name__ == '__main__':
    # make_reviews_csv()
    make_product_csv()
    pass