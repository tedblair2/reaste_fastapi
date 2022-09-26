import Recommender
import Content
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, Form
import uvicorn
import random

app = FastAPI()


@app.get("/")
async def root():
    return {"hello": "world"}


@app.post("/content")
def content(postid: str = Form()):
    # content based filtering to provide recommendations
    results = Content.get_content(postid)
    history = Content.get_history()

    # using collaborative filtering to provide an extra 5 recommendations
    train_data, test_data = train_test_split(history, test_size=0.2, random_state=3)
    model = Recommender.house_recommender(5)
    model.create(train_data, 'userid', 'postid')

    similar = model.similar_items([postid])
    similar_list = list(similar['house_id'])

    results.extend(similar_list)
    results = list(dict.fromkeys(results))

    return {"postlist": results}


@app.post("/collaborative")
def collaborative(userid: str = Form()):
    # using content based filtering to provide personalized recommendations
    user_items = Content.get_user_items(userid)

    results = []
    for item in user_items:
        recommendations = Content.get_content(item)
        results.extend(recommendations)

    results = list(dict.fromkeys(results))
    if len(results) < 10:
        results_random = results
    else:
        results_random = random.sample(results, 10)

    # using collaborative filtering to provide personalized recommendations
    history = Content.get_history()
    users = history['userid'].unique()
    houses = history['postid'].unique().tolist()
    user_list = users.tolist()
    train_data, test_data = train_test_split(history, test_size=0.2, random_state=3)
    model = Recommender.house_recommender(10)
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
