import Recommender
import pandas
import pyrebase
import Content
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI, Form
import uvicorn
import random

app = FastAPI()


def get_houses():
    config = {
        'apiKey': "AIzaSyBqdBMwUd7wp_FioYW_PdaU5iGStTGeJ1w",
        'authDomain': "alvin-9f1e7.firebaseapp.com",
        'databaseURL': "https://alvin-9f1e7.firebaseio.com",
        'projectId': "alvin-9f1e7",
        'storageBucket': "alvin-9f1e7.appspot.com",
        'messagingSenderId': "264584905386",
        'appId': "1:264584905386:web:32911ca6805d8a4f6e46d3"
    }
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()

    columns = ['postid', 'location', 'price', 'bedrooms']
    houses = pandas.DataFrame(columns=columns)
    items = db.child("Posts").get()

    for item in items:
        houses.loc[len(houses)] = [item.val()['postid'], item.val()['location'], item.val()['price'],
                                   item.val()['bedrooms']]

    return houses


def get_history():
    config = {
        'apiKey': "AIzaSyBqdBMwUd7wp_FioYW_PdaU5iGStTGeJ1w",
        'authDomain': "alvin-9f1e7.firebaseapp.com",
        'databaseURL': "https://alvin-9f1e7.firebaseio.com",
        'projectId': "alvin-9f1e7",
        'storageBucket': "alvin-9f1e7.appspot.com",
        'messagingSenderId': "264584905386",
        'appId': "1:264584905386:web:32911ca6805d8a4f6e46d3"
    }
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    columns = ['userid', 'postid']
    history1 = pandas.DataFrame(columns=columns)
    hist = db.child("History").get()

    for item in hist:
        history1.loc[len(history1)] = [item.val()['userid'], item.val()['postid']]

    history = history1.drop_duplicates(keep='first')

    return history


def get_user_items(user):
    history = get_history()
    user_data = history[history['userid'] == user]
    user_items = list(user_data['postid'].unique())

    return user_items


# content based recommendation
def get_recommendations(itemid):
    houses = get_houses()
    house_list = list(houses['postid'])
    id = house_list.index(itemid)

    def get_important_columns(data):
        important_columns = []
        for i in range(0, data.shape[0]):
            important_columns.append(data['location'][i] + '' + str(data['price'][i]) + '' + str(data['bedrooms'][i]))
        return important_columns

    houses['important columns'] = get_important_columns(houses)
    houses.reset_index(inplace=True)
    houses = houses.rename(columns={'index': 'id'})

    # using in content based recommendation for each house
    cm = CountVectorizer().fit_transform(houses['important columns'])
    cs = cosine_similarity(cm)

    scores = list(enumerate(cs[int(id)]))
    sorted_list = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_list = sorted_list[1:]

    final = ['postid', 'location', 'rank']
    df = pandas.DataFrame(columns=final)

    j = 0
    for item in sorted_list:
        location = houses[houses.id == item[0]]['location'].values[0]
        posts = houses[houses.id == item[0]]['postid'].values[0]

        df.loc[len(df)] = [posts, location, j + 1]
        j = j + 1
        if j > 4:
            break
    results = list(df['postid'])

    return results


@app.get("/")
async def root():
    return {"hello": "world"}


@app.post("/content")
def content(postid: str = Form()):
    # content based filtering to provide recommendations
    results = get_recommendations(postid)
    history = get_history()

    # using collaborative filtering to provide an extra 5 recommendations
    train_data, test_data = train_test_split(history, test_size=0.2, random_state=1)
    model = Content.house_recommender()
    model.create(train_data, 'userid', 'postid')

    similar = model.similar_items([postid])
    similar_list = list(similar['house_id'])

    results.extend(similar_list)
    results = list(dict.fromkeys(results))

    return {"postlist": results}


@app.post("/collaborative")
def collaborative(userid: str = Form()):
    # using content based filtering to provide personalized recommendations
    user_items = get_user_items(userid)

    results = []
    for item in user_items:
        recommendations = get_recommendations(item)
        results.extend(recommendations)

    results = list(dict.fromkeys(results))
    results_random = random.sample(results, 10)

    # using collaborative filtering to provide personalized recommendations
    history = get_history()
    users = history['userid'].unique()
    houses = history['postid'].unique().tolist()
    user_list = users.tolist()
    train_data, test_data = train_test_split(history, test_size=0.15, random_state=1)
    model = Recommender.house_recommender()
    model.create(train_data, 'userid', 'postid')

    if userid not in user_list:
        houses_list = random.sample(houses, 10)
        return {"postlist": houses_list}
    else:
        user_results = model.recommend(userid)
        user_results_list = list(user_results['house_id'])

    user_results_list.extend(results_random)
    final_list = list(dict.fromkeys(user_results_list))

    return {"postlist": final_list}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
