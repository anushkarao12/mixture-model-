import numpy as np
import kmeans
import common
import naive_em
import em


X = np.loadtxt("toy_data.txt")

# 1. K-means

def run_kmeans():
    for K in range(1, 5):
        min_cost = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            if min_cost is None or cost < min_cost:
                min_cost = cost
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        title = "K-means for K={}, seed={}, cost={}".format(K, best_seed, min_cost)
        print(title)
        common.plot(X, mixture, post, title)

run_kmeans()

# 2.a Naive EM after e-step

def first_estep():
    K = 3
    mixture, _ = common.init(X, K, seed=0)
    (post, ll) = naive_em.estep(X, mixture)
    print("Log-likelihood: {}".format(ll))

first_estep()


# 2.b Naive EM

def run_naive_em():
    for K in range(1, 5):
        max_ll = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = naive_em.run(X, mixture, post)
            if max_ll is None or ll > max_ll:
                max_ll = ll
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, ll = naive_em.run(X, mixture, post)
        title = "EM for K={}, seed={}, ll={}".format(K, best_seed, ll)
        print(title)
        common.plot(X, mixture, post, title)

run_naive_em()

# 2.c BIC

def run_naive_em_with_bic():
    max_bic = None
    for K in range(1, 5):
        max_ll = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = naive_em.run(X, mixture, post)
            if max_ll is None or ll > max_ll:
                max_ll = ll
                best_seed = seed

        mixture, post = common.init(X, K, best_seed)
        mixture, post, ll = naive_em.run(X, mixture, post)
        bic = common.bic(X, mixture, ll)
        if max_bic is None or bic > max_bic:
            max_bic = bic
        title = "EM for K={}, seed={}, ll={}, bic={}".format(K, best_seed, ll, bic)
        print(title)
        common.plot(X, mixture, post, title)

run_naive_em_with_bic()


# 3.a Matrix completion

X = np.loadtxt("netflix_incomplete.txt")

def run_em_netflix():
    for K in [1, 12]:
        max_ll = None
        best_seed = None
        for seed in range(0, 5):
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = em.run(X, mixture, post)
            if max_ll is None or ll > max_ll:
                max_ll = ll
                best_seed = seed

        title = "EM for K={}, seed={}, ll={}".format(K, best_seed, max_ll)
        print(title)

run_em_netflix()


def run_matrix_completion():
    K = 12
    seed = 1
    mixture, post = common.init(X, K, seed)
    mixture, post, ll = em.run(X, mixture, post)
    X_pred = em.fill_matrix(X, mixture)
    X_gold = np.loadtxt('netflix_complete.txt')
    print("RMSE:", common.rmse(X_gold, X_pred))

run_matrix_completion()

