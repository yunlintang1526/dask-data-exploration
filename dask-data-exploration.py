import dask
import dask.dataframe as dd
import dask.array as da
import numpy as np
import json
import pandas as pd
from distributed import Client

def Assignment1A(user_reviews_csv):
    client = Client('127.0.0.1:8786')
    client = client.restart()    
    
    def review_year(time):
        return int(time[-4:])

    def clean_helpful(x):
        x = x.strip('[').strip(']')
        return x.split(',')
    
    def pos_vote(vote):
        vote = clean_helpful(vote)
        return int(vote[0])

    def total_vote(vote):
        vote = clean_helpful(vote)
        return int(vote[1])
    
    reviews = dd.read_csv(user_reviews_csv,
                          dtype={'overall':'float64'},
                          usecols=['reviewerID', 'helpful', 'overall', 'reviewTime'])
    
    reviews['reviewTime']=reviews['reviewTime'].apply(review_year,meta=('int'))
    reviews['positive']=reviews['helpful'].apply(pos_vote, meta=('int'))
    reviews['total']=reviews['helpful'].apply(total_vote, meta=('int'))
    
    computations = reviews.groupby('reviewerID').agg({'reviewerID':'count',
                                        'overall':'mean',
                                        'reviewTime':'min',
                                        'positive':'sum',
                                        'total':'sum'},split_out=4).compute()
    com_val = computations.values
    
    temp = pd.DataFrame(columns=['reviewerID','number_products_rated', 'avg_ratings', 'reviewing_since',
       'helpful_votes', 'total_votes'])
    temp['reviewerID'] = computations.index
    float_cols = ['number_products_rated', 'avg_ratings', 'reviewing_since', 'helpful_votes', 'total_votes']
    temp[float_cols] = com_val
    temp[float_cols] = temp[float_cols].astype(np.float64)
    
    mem = reviews.memory_usage(deep=True).sum().compute()
    n = int(1 + mem//100e6);
    df = dd.from_pandas(temp, npartitions=n)
    
    
    submit = df.describe().compute().round(2)    
    with open('results_1A.json', 'w') as outfile: json.dump(json.loads(submit.to_json()), outfile)


def Assignment1B(user_reviews_csv,products_csv):
    client = Client('127.0.0.1:8786')
    client = client.restart()
    
    reviews = dd.read_csv(user_reviews_csv)
    products = dd.read_csv(products_csv)
    
    # problem 1
    temp_reviews = reviews.isnull().sum().compute()
    temp_products = products.isnull().sum().compute()
    b1_reviews = (temp_reviews * 100 / len(reviews)).round(2)
    b1_products = (temp_products * 100 / len(products)).round(2)
    
    # store q1 into result dict
    result={}
    result['q1']={}
    result['q1']['products'] = dict(b1_products)
    result['q1']['reviews'] = dict(b1_reviews)
    
    # problem 2
    r = reviews[['asin', 'overall']]
    p = products[['asin', 'price']]
    join_rp = dd.merge(r, p, on='asin', how='inner').corr().compute()
    b2 = join_rp['overall']['price'].round(2)
    
    # store q2 into result dict
    result['q2'] = b2
    
    # problem 3
    statistics = products['price'].describe().compute()
    b3 = statistics[['mean', 'std', '50%', 'min', 'max']].round(2)
    
    # store q3 into result dict
    result['q3'] = dict(b3)
    
    # problem 4
    def super_cat(cat):
        try:
            return eval(cat)[0][0]
        except:
            return cat
    
    cat = products.categories.apply(super_cat, meta=('categories', 'object')).compute()
    b4 = cat.value_counts()
    
    # store q4 into result dict
    b4_dict = dict(b4)
    for key in b4_dict.keys():
        if key == '':
            b4_dict[key] = int(b4_dict[key]) + int(temp_products['categories'])
        else:
            b4_dict[key] = int(b4_dict[key])
    result['q4'] = b4_dict
    
    # problem 5
    re=reviews.asin.unique().compute()
    pr=products.asin.unique().compute()
    b5 = len(re.to_frame().merge(pr.to_frame(), on='asin', how='inner')) < len(re)
    
    # store q5 into result dict
    result['q5'] = int(b5)
    
    # problem 6
    def clean(x):
        try:
            return list(eval(x).values())
        except:
            return np.nan
    
    clean_related = products.related.apply(clean, meta=('related','object'))
    flatten = clean_related.explode().explode()
    
    b6 = 0
    check_list = list(products['asin'].unique())
    for ele in flatten:
        if ele not in check_list:
            b6 = 1
            break
    
    # store q6 into result dict
    result['q6'] = b6

    # Write your results to "results_1B.json" here and round your solutions to 2 decimal points
    with open('results_1B.json', 'w') as outfile: json.dump(result, outfile)