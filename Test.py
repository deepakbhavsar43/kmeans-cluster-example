import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class kmeans:

    def __init__(self, n_cluster=3, max_iter = 300):
        self.n_cluster = n_cluster
        self.max_iter = max_iter


    def min_max(self):
        xmin = min(df['X'])
        ymin = min(df['Y'])
        xmax = max(df['X'])
        ymax = max(df['Y'])
        return

df = pd.DataFrame({
    'X': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'Y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})


# print()
# exit()

k = 3

# centroid ={}
# for i in range(k):
#     centroid[i+1] = [np.random.randint(0, 80), np.random.randint(0, 80)]

centroids = {
    i + 1: [np.random.randint(0, 80), np.random.randint(0, 80)] for i in range(k)
}

# To visualize data-points and initial centroids
# plt.scatter(df["X"], df["Y"])
colmap = {1: 'red', 2: 'green', 3: 'blue'}
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()


# exit()
# print(df['X'].head())
# print("--------------")

def assignment1(df, centroids):
    for i in centroids.keys():
        df[f'distance_from_{i}'] = (
            np.sqrt(
                (df['X'] - centroids[i][0]) ** 2 + (df['Y'] - centroids[i][1]) ** 2
            )
        )
    # print(df.head())
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    # print(centroid_distance_cols)
    return df


df = assignment1(df, centroids)
# print("assignment\n", df)

# print(df.head())
plt.scatter(df['X'], df['Y'], color=df["color"])
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='X')
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

old_cent = centroids.copy()
print(centroids)
# print("update")
def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['X'])
        # print(df[df['closest'] == i]['X'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['Y'])
        # print(np.mean(df[df['closest'] == i]['Y']))
    return k


centroids = update(centroids)
# print(centroids)

# fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['X'], df['Y'], color=df['color'])
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], marker='x')
plt.xlim(0, 80)
plt.ylim(0, 80)
# for i in old_cent.keys():
#     old_x = old_cent[i][0]
#     old_y = old_cent[i][1]
#     dx = (centroids[i][0] - old_cent[i][0] * 0.75)
#     dy = (centroids[i][1] - old_cent[i][0] * 0.75)
#     ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()

df = assignment1(df, centroids)
# fig = plt.figure(figsize=(5, 5))
plt.scatter(df['X'], df['Y'], color=df['color'], label='Datapoints')
for i in centroids.keys():
    plt.scatter(*centroids[i], marker='x', label="Centroids")
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.legend()
plt.show()

