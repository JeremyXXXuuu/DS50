import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pymongo

def load_from_mongo():
    client = pymongo.MongoClient("mongodb+srv://jeremy:root@cluster0.5ei45.mongodb.net/test")
    db = client.database
    collection = db['steam_user']
    df= pd.DataFrame(list(collection.find()))
    del df['_id']
    return df

#Make predictions based on cosine similarities
def predict(ratings, similarity, type='user' ):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred


#Find the difference between the games played for the target and the referred user
#To avoid recommending for user with games that have already been played
def find_diff(userid1,userid2):
    game1=clean_data[clean_data["userid"]==userid1]["gameIdx"].tolist()
    game2=clean_data[clean_data["userid"]==userid2]["gameIdx"].tolist()
    diff = set(game2).difference(set(game1))
    return diff

#Sort the games according to their scores and recommend the best one
def sort_game(game_diff,useridx,similar_indices,num):
    if len(game_diff) < num:        #Pour éviter de recommander pour objectif avec des jeux qui ont déjà été joués
        num=len(game_diff)
    game_rating=[None]*len(game_diff)
    for i in range(len(game_diff)):
        game_rating[i]=user_game_interactions[similar_indices[0]][list(game_diff)[i]]
    game_rating=np.array(game_rating)
    sort=game_rating.argsort()[::-1]
    recomand_game=[0]*num
    for i in range(num):  
#        print(idx2game[list(game_diff)[sort[i]]])
#        print(clean_data.loc[clean_data['gameIdx']==list(game_diff)[sort[i]]]['appid'].tolist()[0])
        recomand_game[i]=clean_data.loc[clean_data['gameIdx']==list(game_diff)[sort[i]]]['appid'].tolist()[0]
    return recomand_game

#elimination of duplicates
def del_dup(items):
    s = set()
    for item in items:
        if item not in s:
            yield item
            s.add(item)

            
def recommendation(userid,prediction,num):
    total=num
    idx=user2idx[userid]      #Convert user ID to index
    similar_indices = user_prediction[idx].argsort()[::-1]       #Find the most similar user
    userid2=idx2user[similar_indices[0]]
    game_diff=find_diff(userid,userid2)     #Find the different games of both users
    #If the most similar user has less games different from those of the target
    #Find other two similar users
    if len(game_diff)>=num:     
        sort_game(game_diff,similar_indices[0],similar_indices,num)      #Use sort_game to make the recommendation
        num=0
    else:
        i=0
        #Find up to three similar users to recommend
        recommended=[]  #Previously recommended games
        while num>0 & i<3:
            userid2=idx2user[similar_indices[i]]
            game_diff=find_diff(userid,userid2)
            num_rec=num-len(game_diff)
            if len(game_diff)!=0:
                recomand_game=sort_game(game_diff,similar_indices[i],similar_indices,num)
                for i in range(len(recomand_game)):
                    recommended.append(recomand_game[i])
            num=num_rec
            i=i+1
    
    #elimination of duplicates
    recommended=list(del_dup(recommended))
    
    #If the number of games that can be recommended by similar users is less than the number of games that should be recommended, 
    #the best selling games will be recommended.
    #Do not recommend games that have already been recommended.   
    if num>0:            
        if len(recommended)!=0:
            for j in range(len(recommended)):
                for i in range(len(topgames)):
                    if recommended[j] == topgames[i][0]:
                        del topgames[i]
        for k in range(num):
            recommended.append(topgames[k])
    #Displays the recommended game ID
    for x in recommended:
        print(x,end=' ')



clean_data = load_from_mongo()
# clean_data = pd.read_csv('D:\Code\DS50\datasets\\user_data.csv',index_col=0)
n_users = len(clean_data.userid.unique()) #Number of users
n_games = len(clean_data.game.unique())   #Number of games

# Create dictionaries to convert user and games into indexes and vice versa
user2idx = {user: i for i, user in enumerate(clean_data.userid.unique())}
idx2user = {i : user for user, i in user2idx.items()}

game2idx = {game: i for i, game in enumerate(clean_data.game.unique())}
idx2game = {i: game for game, i in game2idx.items()}

#Convert the user and the games into an index
user_idx = clean_data['userid'].apply(lambda x: user2idx[x]).values
game_idx = clean_data['gameIdx'] = clean_data['game'].apply(lambda x: game2idx[x]).values
rating = clean_data['rating'].values

# Create a zero matrix
zero_matrix = np.zeros(shape = (n_users, n_games))
user_game_interactions = zero_matrix.copy()
# Fill the matrix with scores
user_game_interactions[user_idx, game_idx] = rating

user_similarity = pairwise_distances(user_game_interactions, metric='cosine')

user_prediction = predict(user_game_interactions, user_similarity, type='user')

#best-selling games
topgames=[0]*10
topgames[0]=[570,"Dota 2"]
topgames[1]=[440,"Team Fortress 2"]
topgames[2]=[304930,"Unturned"]
topgames[3]=[550,"Left 4 Dead 2"]
topgames[4]=[240,"Counter-Strike Source"]
topgames[5]=[10,"Counter-Strike"]
topgames[6]=[4000,"Garry's Mod"]
topgames[7]=[8930,"Sid Meier's Civilization V"]
topgames[8]=[301520,"Robocraft"]
topgames[9]=[320,"Half-Life 2 Deathmatch"]


# input
# userid = int(sys.argv[1])

recommendation(6717871,user_prediction,20)

# output
# output = recommendation(userid, user_prediction, 20)
# sys.stdout.write(output)


