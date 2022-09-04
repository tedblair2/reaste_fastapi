import pyrebase
import pandas
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


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


def get_important_columns(data):
    important_columns = []
    for i in range(0, data.shape[0]):
        important_columns.append(data['location'][i] + '' + str(data['price'][i]) + '' + str(data['bedrooms'][i]))
    return important_columns


def get_content(itemid):
    houses = get_houses()
    index = houses.index[houses['postid'] == itemid].tolist()
    id = index[0]

    houses['important columns'] = get_important_columns(houses)

    # using in content based recommendation for each house
    cm = CountVectorizer().fit_transform(houses['important columns'])
    cs = cosine_similarity(cm)

    scores = list(enumerate(cs[int(id)]))
    sorted_list = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_list = sorted_list[1:]

    columns = ['postid', 'location', 'score']
    df = pandas.DataFrame(columns=columns)

    j = 0
    for item, score in sorted_list:
        posts = houses.at[item, 'postid']
        location = houses.at[item, 'location']

        df.loc[len(df)] = [posts, location, score]
        j = j + 1
        if j > 4:
            break
    final_df = df[df['score'] > 0]
    results = list(final_df['postid'])

    return results
