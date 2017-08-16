import pandas as pd
import numpy as np

# LOAD TWITTER DATA
twitter = pd.read_csv("./twitter_combined.txt", header= None)
twitter["id"] = twitter.iloc[:, 0].apply(lambda x: x.split(' ')[0])
twitter["follows"] = twitter.iloc[:, 0].apply(lambda x: x.split(' ')[1])
twitter = twitter[['id', 'follows']]

# TAKE A SUBSAMPLE
twitter_subsample = twitter[:50000]
edges = dict(twitter_subsample.groupby(['id']).apply(lambda x: list(x['follows'])))
nodes = list(set(list(twitter_subsample.id.unique()) + list(twitter_subsample.follows.unique())))

# CREATE THE ADJACENCY MATRIX
# Create n x n matrix, 1 if row i follows column j
A = []
t = 0
for i in nodes:
    following = []
    for j in nodes:
        # if person i follows anyone
        if i in edges:
            if j in edges[i]:
                following.append(1)
            else:
                following.append(0)
        # if person i doesn't follow anyone (DANGLING NODES)
        else:
            following.append(1) # connect to all other nodes
    A.append(following)
    t += 1


A = np.array(A)

# MAKE A COLUMN STOCHASTIC
A2 = []
for i in range(len(A)):
    sum_ = sum(A[i])
    if sum_ > 0:
        A2.append(A[i]  / sum_)
    else:
        A2.append(A[i])

# CREATE M - GOOGLE MATRIX
A2 = np.array(A2).T
S = np.ones((len(A2), len(A2)))*(1/len(A2))
p = .15
M = (1 - p)*A2 + p*S

# CHECK CONVERGENCE
n = len(A2)
v = (np.ones(len(A2))*(1/len(A2))).reshape(len(A2), 1)
x = v
x_prev = v
i = 0
diffs = []
while True:
    x_prev = x
    x = M.dot(x) / sum(M.dot(x)) # L1 norm
    error = sum(abs(x - x_prev))/float(n)
    if error < 1e-15: # Check convergence
        break
    i += 1
    diffs.append(error)

import matplotlib.pyplot as plt
plt.plot(diffs)
plt.xlabel("Time")
plt.ylabel("Convergence Error")
plt.title("PageRank Vector Approximation Error")
plt.show()

# CHECK OUT THE RANKS
# Sorted in increasing order
x = [i[0] for i in x]
sorted_x = np.argsort(x)

# Index of worst 10, least popular people
worst_idx = sorted_x[:10]
# Index of best 10, most popular people
best_idx = sorted_x[-10:]

worst = np.array(nodes)[worst_idx]
best = np.array(nodes)[best_idx]

print(worst)
print(best)


