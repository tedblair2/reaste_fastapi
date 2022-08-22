import Recommender
import pandas
import pyrebase
import Content
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI, Form
import uvicorn

app = FastAPI()


def data():
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


def data2():
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


@app.get("/")
async def root():
    return {"hello": "world"}


@app.post("/content")
def content(postid: str = Form()):
    dataset = data()
    dataset2 = data2()

    house_list = list(dataset['postid'])
    id = house_list.index(postid)

    def get_important_columns(data):
        important_columns = []
        for i in range(0, data.shape[0]):
            important_columns.append(data['location'][i] + '' + str(data['price'][i]) + '' + str(data['bedrooms'][i]))
        return important_columns

    dataset['important columns'] = get_important_columns(dataset)
    dataset.reset_index(inplace=True)
    dataset = dataset.rename(columns={'index': 'id'})

    # using in content based recommendation for each house
    cm = CountVectorizer().fit_transform(dataset['important columns'])
    cs = cosine_similarity(cm)

    scores = list(enumerate(cs[int(id)]))
    sorted_list = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_list = sorted_list[1:]

    final = ['postid', 'location', 'rank']
    df = pandas.DataFrame(columns=final)

    j = 0
    for item in sorted_list:
        location = dataset[dataset.id == item[0]]['location'].values[0]
        posts = dataset[dataset.id == item[0]]['postid'].values[0]

        df.loc[len(df)] = [posts, location, j + 1]
        j = j + 1
        if j > 4:
            break
    results = list(df['postid'])

    # using collaborative filtering to provide an extra 5 recommendations
    train_data, test_data = train_test_split(dataset2, test_size=0.2, random_state=1)
    model = Content.house_recommender()
    model.create(train_data, 'userid', 'postid')

    similar = model.similar_items([postid])
    similar_list = list(similar['house_id'])

    results.extend(similar_list)
    results = list(dict.fromkeys(results))

    return {"postlist": results}


@app.post("/collaborative")
def collaborative(userid: str = Form()):
    history = data2()
    users = history['userid'].unique()
    user_list = users.tolist()
    train_data, test_data = train_test_split(history, test_size=0.2, random_state=1)
    model = Recommender.house_recommender()
    model.create(train_data, 'userid', 'postid')

    error = "nothing"
    if userid not in user_list:
        return {"postlist": [error]}
    else:
        position = user_list.index(userid)
    user_id = users[position]
    user_results = model.recommend(user_id)
    user_results_list = list(user_results['house_id'])

    return {"postlist": user_results_list}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
