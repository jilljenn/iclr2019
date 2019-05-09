from sklearn.linear_model import SGDClassifier
from scipy.sparse import load_npz, save_npz
import numpy as np
import random


TOP = 3


def display_and_ask(clf):
    proba = list(clf.predict_proba(X)[:, 1])
    pivot = min((abs(proba[i] - 0.5), i) for i, p in enumerate(proba))[1]
    increasing = list(np.argsort(proba))
    n = len(increasing)
    print('= NO =')
    for i in increasing[:TOP]:
        print(corpus[i], proba[i])
    print('= CONTROVERSIAL =')
    for i in increasing[pivot - TOP // 2:pivot + TOP // 2]:
        print(corpus[i], proba[i])
    print('= YES =')
    for i in increasing[-TOP:]:
        print(corpus[i], proba[i])
    return pivot


with open('tfidf.txt') as f:
    corpus = f.read().splitlines()
X = load_npz('tfidf.npz')
print(X.shape)


clf = SGDClassifier(max_iter=1000, tol=1e-3, loss='log')
asked = [random.choice(list(range(len(corpus))))]
y = []
while True:
    print()
    answer = input(corpus[asked[-1]] + '? ')
    y.append(1 if answer == '1' else 0)
    print(asked)
    clf.partial_fit(X[asked, :], y, [0, 1])
    pivot = display_and_ask(clf)
    asked.append(pivot)
