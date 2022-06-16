import pandas as pd
import numpy as np
import heapq
import sys
import pymongo



def load_from_mongo():
    client = pymongo.MongoClient("mongodb+srv://jeremy:root@cluster0.5ei45.mongodb.net/database?retryWrites=true&w=majority")
    db = client.database
    collection = db['steam_games2']
    df= pd.DataFrame(list(collection.find()))
    del df['_id']
    return df



# Translate game id to index
def game_id2idx(game_id):
    idx = steam[steam["app_id"]== game_id].index.tolist()[0]
    return idx



#Calculez la similarité d'autres jeux avec ce jeu.
#Un point pour une étiquette identique
def get_score(game_id):
    game_index=game_id2idx(game_id)
    score = [0]*17282
    for i in range(len(tags[game_index])):
        for indexs in steam.index:
            if(indexs != game_index):
                for j in range(len(tags[indexs])):
                    if tags[game_index][i] == tags[indexs][j]:
                        score[indexs] = score[indexs]+1
    return score



#Trouvez le nom du jeu et de l'indice les plus élevés.
def get_index(game_id,num):
    score=get_score(game_id)
    re1 = heapq.nlargest(num, score)
    #print(re1)
    re2=[0]*num
    for i in range(len(re1)):    
        re2[i] = score.index(re1[i])
        if i>0 and re2[i] == re2[i-1]:
            score[re2[i]]=0
            re2[i] = score.index(re1[i])
    #print(re2)  
    return re1,re2



def recommend(game_id,num):
    re1,re2=get_index(game_id,num)
    #print("Recommended games:")
    id_list = []
    for i in range(len(re1)):
        game_id=steam.loc[re2[i],"app_id"]
        id_list.append(int(game_id))
        #print("<{0}> with score:{1}\n".format(name,re1[i]))
    print(id_list)




# steam = load_from_mongo()
steam = pd.read_csv("https://raw.githubusercontent.com/alyang666/DS50/main/datastes/steam_games2.csv")

#steam = steam.iloc[0:,1:8]
#steam = steam.drop(index = steam[(steam.types == "bundle")].index.tolist())
#steam = steam.drop(index = steam[(steam.types == "sub")].index.tolist())
tags=[None]*len(steam)

for indexs in steam.index:
    str = steam.loc[indexs,"popular_tags"]
    if isinstance(str,float):
        #print(str)
        steam = steam.drop(index = indexs)
        indexs = indexs -1
    else:
        str = str.split(',')
        tags[indexs] = str

# Input from back-end nodejs
game_name = int(sys.argv[1])
num = int(sys.argv[2])

# print(recommend(game_name, num))
# Output to the backend
output = recommend(game_name, num)
sys.stdout.write(output)




