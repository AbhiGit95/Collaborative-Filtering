import pandas as pd
import numpy as np
import time
import gc
start = time.time()

train_df = pd.read_csv("netflix/TrainingRatings.txt", header = None)
train_df.columns = ["movie_id", "user_id", "rating"]

test_df = pd.read_csv("netflix/TestingRatings.txt", header = None)
test_df.columns = ["movie_id", "user_id", "rating"]

#unique_users = list(train_df["user_id"].unique())

bdf = pd.DataFrame()
for movie, group in train_df.groupby("movie_id"):
    if bdf.empty:
        bdf = group.set_index("user_id")[["rating"]].rename(columns= {"rating" : movie})
    else:
        bdf = bdf.join(group.set_index("user_id")[["rating"]].rename(columns= {"rating" : movie}), how= "outer")

bad = (3594, 4256, 8570, 13074, 14354, 16309, 16631, 18291, 19183, 23730, 24696, 25203, 26756)
bad = list(bad)
bu = [list(bdf.index)[b] for b in list(bad)]
bdf.drop(bu, inplace = True)


bdf.fillna(0)
r_arr = bdf.values
r_arr[np.isnan(r_arr)] = 0.0


r_sum = r_arr.sum(axis = 1)
r_nz = np.count_nonzero(r_arr, axis = 1)
r_avg = np.true_divide(r_sum, r_nz)

r_sub = r_arr - r_avg.reshape(bdf.shape[0], 1)
tv = r_arr == 0
r_sub[tv] = 0
r_num = np.dot(r_sub, r_sub.T)


gc.collect()


a_sqr_d = np.square(r_sub).sum(axis = 1)
W = np.multiply(a_sqr_d.reshape(bdf.shape[0], 1), a_sqr_d.reshape(bdf.shape[0], 1).T)
W = np.sqrt(W)
W = np.true_divide(r_num, W)



paj = []
for i, row in test_df[["movie_id", "user_id"]].iterrows():
    if row["user_id"] in bu:
        paj.append(2.5)
    else:
        a = list(bdf.index).index(row["user_id"])
        j = list(bdf.columns).index(row["movie_id"])
        k = np.true_divide(1, W[a].sum())
        tp = r_avg[a] + np.multiply(k, np.dot(W[a], r_sub[:, j]))
        paj.append(tp)

paj = np.array(paj)
Y = test_df["rating"].values
error = np.true_divide(np.sum(np.abs(Y - paj)), len(Y))
sqerror = np.sqrt(np.true_divide(np.sum(np.square(Y - paj)) , len(Y)))

print("Time taken = {}".format(time.time() - start))
