from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd

product_info = pd.read_csv('CatfoodProductInfo.csv')
reviews_info = pd.read_csv('CatfoodReviewsInfo.csv')

dbname = 'catfood_db'
username = 'danielben-zion'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print(engine.url)

if not database_exists(engine.url):
    create_database(engine.url)

product_info.to_sql('product_info_table', engine, if_exists='replace')
reviews_info.to_sql('reviews_table', engine, if_exists='replace')