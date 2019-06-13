import glob
import pickle
import re
import csv
import os
from bs4 import BeautifulSoup

basedir = '/Users/danielben-zion/Dropbox/insight/teefies/scraping'

all_links = pickle.load( open('links.pkl','rb') )


def make_reviews_csv():
    # Reviews CSV file
    columns = ['product', 'review_author','rating','review_text','helpful_rank']
    with open('/Users/danielben-zion/Dropbox/insight/teefies/scraping/CatfoodReviewsInfo.csv','w') as catfoodcsv:
        writer = csv.writer(catfoodcsv)
        writer.writerow(columns)


        review_counter = 0
        product_counter = 0 
        for (product_name, product_link) in all_links.items():
            

            product_dir = product_name.split(',')[0].replace(' ','-').lower()

            # print(product_dir)
            for reviewpage in glob.glob(f'{basedir}/html_pages/individual-items/{product_dir}/page*.html'):

                # print(reviewpage)

                review_soup = BeautifulSoup(open(reviewpage,'r'),'html')

                for review in review_soup.find_all("li", {'itemprop': 'review'}):
                    rating = review.find('meta',{'itemprop':'ratingValue'})['content']
                    review_text = review.find('span',{'class' : 'ugc-list__review__display'}).text
                    review_author = review.find('span',{'itemprop' : 'author'}).text
                    helpful = review.find('button',{'class' : 'cw-btn cw-btn--white cw-btn--sm ugc__like js-like'}).span.text
                    writer.writerow([product_name,review_author,rating,review_text,helpful])
                    review_counter += 1

            product_counter += 1
        print(f'Succesfully made csv file for {review_counter} reviews across {product_counter} products')



def make_product_csv():
    # Product Info CSV file
    columns = ['product','brand', 'price','oz_per_can','num_cans','price_per_oz']
    with open('/Users/danielben-zion/Dropbox/insight/teefies/scraping/CatfoodProductInfo.csv','w') as catfoodcsv:
        writer = csv.writer(catfoodcsv)
        writer.writerow(columns)


        counter = 0 
        for (product_name, product_link) in all_links.items():
            product_dir = product_name.split(',')[0].replace(' ','-').lower()

            main_page_soup = BeautifulSoup(open(f'{basedir}/html_pages/individual-items/{product_dir}/main_page.html','r'),'html')

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
            
            
            brand = main_page_soup.find('span',{'itemprop' : 'brand'}).text
            

    #         print(f'Full product name: {product_name}')

    #         print(f"Name: {nn}" )
    #         print(f'oz_per_can: {opc}')
    #         print(f'num_cans: {ncans}')
    #         print(f'price: {price}')
            

            row = [product_name,brand,price,oz_per_can,num_cans,ppo]
            writer.writerow(row)
            counter += 1

        print(f'Succesfully made csv data file for {counter} products')


if __name__ == '__main__':
    # make_reviews_csv()
    make_product_csv()