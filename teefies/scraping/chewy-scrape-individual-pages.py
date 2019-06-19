from bs4 import BeautifulSoup
from selenium import webdriver
import os
import time
import pickle
import glob
import string


basedir = '/Users/danielben-zion/Dropbox/insight/teefies/data'


def setOptions():
    options = webdriver.FirefoxOptions()
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
#     options.add_argument('headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--no-proxy-server')
    return options

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


# Now is a good time to activate some kind of IP switching software
pages = range(1,10)
succeed_counter = 0
fail_counter = 0
variety_counter = 0
options = setOptions()

all_links = pickle.load( open('links.pkl','rb') )
print(len(all_links))

with webdriver.Firefox(options=options) as driver:
    for (product_id, product_main_link) in all_links.items():
        try:
            name = product_id.translate(str.maketrans('', '', string.punctuation))
            
            name = name.replace(' ','-').lower()
            filepath = os.path.join(basedir,f'html_pages/individual-items/{name}/')
    #             print(filepath)

            

            

            filepathname = f'{filepath}/main_page.html'
            
#             if BeautifulSoup(open(filepathname,'r'),'html').find('html'):
#                 counter += 1
#                 continue
                
            driver.get(str(product_main_link))
            time.sleep(2)
            main_page_soup = BeautifulSoup(driver.page_source,'html')

            if 'Variety' in main_page_soup.find('div', {'id' : 'attributes'}).text:
                variety_counter +=1
                continue


            ensure_dir(filepath)
            
            with open(filepathname,'w') as file:
                file.write(driver.page_source)

            
            review_url_init = main_page_soup.find('a',{'class' : 'cw-btn cw-btn--default',
              'href': True})['href']

            review_url_stem = review_url_init.replace('NEWEST','MOST_RELEVANT')[0:-1]
    #         print(review_url_stem)

            for pagenum in pages:
                filepathname = f'{filepath}/page{pagenum}.html'
                
#                 if BeautifulSoup(open(filepathname,'r'),'html').find('html'):
#                     continue
                
                review_url = 'https://www.chewy.com'+review_url_stem+ str(pagenum)
#                 print(f'getting {review_url}')
                driver.get(review_url)
                time.sleep(2)

                with open(filepathname,'w') as file:
                    file.write(driver.page_source)




            succeed_counter+=1   
        except:
            fail_counter +=1
            continue

print(f'Succeeded: {succeed_counter}, Failed: {fail_counter}, Variety Packs Skipped: {variety_counter}')