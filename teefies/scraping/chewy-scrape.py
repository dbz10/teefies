from bs4 import BeautifulSoup
from selenium import webdriver
import os
import time
import pickle
import glob

basedir = '/Users/danielben-zion/Dropbox/insight/teefies/scraping'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        



# scrape full product pages
pagenums = range(1,10)
with webdriver.Firefox() as driver:
    for pagenum in pagenums:
        driver.get(f'https://www.chewy.com/s?rh=c%3A288%2Cc%3A332%2Cc%3A293&&page={pagenum}&sort=popularity')
        time.sleep(5)
        
        filepathname = f'{basedir}/html_pages/main-page/dogs/page{pagenum}.html'
        
        with open(filepathname,'w') as file:
            file.write(driver.page_source)
            

# go through saved multi-product pages, extract link to individual products. 
# ignore products which have multi, variety, medley, etc in the title
# later, we can further filter out 'variety pack' packaging type from main page. 