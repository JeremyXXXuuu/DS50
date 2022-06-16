import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

def load_from_mongo():
    client = pymongo.MongoClient("mongodb+srv://jeremy:root@cluster0.5ei45.mongodb.net/database?retryWrites=true&w=majority")
    db = client.database
    collection = db['steam_reviews_cum']
    df= pd.DataFrame(list(collection.find()))
    del df['_id']
    return df

def item(id):
  return reviews.loc[reviews['app_id'] == id]['app_name'].tolist()[0]  #itemid to name


#Recommander les jeux les plus similaires
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    idx = reviews[reviews["app_id"]== item_id].index.tolist()[0] #itemid to index
    similar_indices = cosine_similarities[idx].argsort()[::-1]  #sort the array in ascending order and return the index
    for i in range(num):
       print("Recommended: " + item(reviews["app_id"][similar_indices[i+1]]) + " (score:" +      str(cosine_similarities[idx][similar_indices[i+1]]) + ")")


# reviews = load_from_mongo()
reviews = pd.read_csv('https://raw.githubusercontent.com/alyang666/DS50/main/datastes/steam_reviews_cum.csv')
steam = pd.read_csv("https://raw.githubusercontent.com/alyang666/DS50/main/datastes/gameids.csv")
arrs=steam['appid'].values
reviews=reviews.loc[reviews['app_id'].isin(arrs)]
reviews=reviews.reset_index(drop=True)

#Évaluation des critiques de jeux à l'aide du TfidfVectorizer de Sklearn.
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(reviews['review_cum'])

#Calcul de la similarité
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
data1 = pd.DataFrame(cosine_similarities)
data1.to_csv('cosine_similarities.csv')

# recommend(item_id=10, num=2)













