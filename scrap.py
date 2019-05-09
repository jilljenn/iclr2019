from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDClassifier
from scipy.sparse import load_npz, save_npz
import numpy as np


with open('accepted.html') as f:
    html = f.read()

b = BeautifulSoup(html)
vectorizer = TfidfVectorizer()
corpus = []
for poster in b.select('div.maincard.Poster'):
    date = ' '.join(element.get_text() for element in
                    poster.select('.maincardHeader'))
    title = poster.select_one('.maincardBody').get_text()
    authors = [''.join(filter(str.isalpha, author)) for author in
               poster.select_one('.maincardFooter').get_text().split(' · ')]
    sentence = title + ' ' + ' '.join(authors)
    corpus.append(sentence)

with open('tfidf.txt', 'w') as f:
    f.write('\n'.join(corpus))

X = vectorizer.fit_transform(corpus)
save_npz('tfidf.npz', X)

print('Closest neighbors TF-IDF')
best = np.argsort(cosine_similarity(X[0], X)).reshape(-1)[::-1]
for test in best[:5]:
    print(test)
    print(corpus[test])
