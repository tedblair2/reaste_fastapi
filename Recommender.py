import pandas
import numpy


class house_recommender():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.houses_dict = None
        self.rev_houses_dict = None
        self.item_similarity_recommendations = None

    # we get unique items corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())

        return user_items

    # we get unique users for a given item
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())

        return item_users

    # we get unique items in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())

        return all_items

    def construct_cooccurence_matrix(self, user_houses, all_houses):

        # we get users for each of the user_houses.
        user_houses_users = []
        for i in range(0, len(user_houses)):
            user_houses_users.append(self.get_item_users(user_houses[i]))

        # Initialize the item cooccurence matrix of size len(user_houses) X len(houses)
        cooccurence_matrix = numpy.matrix(numpy.zeros(shape=(len(user_houses), len(all_houses))), float)

        # we calculate similarity between user houses and all unique houses
        for i in range(0, len(all_houses)):
            # we get unique users of house i
            houses_i_data = self.train_data[self.train_data[self.item_id] == all_houses[i]]
            users_i = set(houses_i_data[self.user_id].unique())

            for j in range(0, len(user_houses)):

                # we get unique users of house j
                users_j = user_houses_users[j]

                users_intersection = users_i.intersection(users_j)

                # we use jaccard similarity to get similarity measure btn house i and j
                if len(users_intersection) != 0:

                    users_union = users_i.union(users_j)

                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        return cooccurence_matrix

    def generate_top_recommendations(self, user, cooccurence_matrix, all_houses, user_houses):
        print("Non zero values in cooccurence_matrix :%d" % numpy.count_nonzero(cooccurence_matrix))

        # Calculate a weighted average of the scores in cooccurence matrix for all user houses.
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = numpy.array(user_sim_scores)[0].tolist()

        # Sort the indices of user_sim_scores based upon their value
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)

        columns = ['user_id', 'house_id', 'score', 'rank']

        df = pandas.DataFrame(columns=columns)

        rank = 1
        for i in range(0, len(sort_index)):
            if ~numpy.isnan(sort_index[i][0]) and all_houses[sort_index[i][1]] not in user_houses and rank <= 15:
                df.loc[len(df)] = [user, all_houses[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        if df.shape[0] == 0:
            print("The current user has no history for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def recommend(self, user):

        user_houses = self.get_user_items(user)

        print("No. of unique houses for the user: %d" % len(user_houses))

        # Get all unique items (houses) in the training data
        all_houses = self.get_all_items_train_data()

        print("no. of unique houses in the training set: %d" % len(all_houses))

        cooccurence_matrix = self.construct_cooccurence_matrix(user_houses, all_houses)

        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_houses, user_houses)

        return df_recommendations

    def similar_items(self, item_list):

        user_houses = item_list

        # Get all unique items (houses) in the training data
        all_houses = self.get_all_items_train_data()

        print("no. of unique houses in the training set: %d" % len(all_houses))

        # Construct item cooccurence matrix of size
        # len(user_houses) X len(houses)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_houses, all_houses)

        # Use the cooccurence matrix to make recommendations
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_houses, user_houses)

        return df_recommendations
