# Recommender system of ICLR 2019 posters

1. Scrape [all accepted ICLR 2019 papers](https://www.iclr.cc/Conferences/2019/Schedule?type=Poster)
1. Extract embeddings from poster titles + authors (+ abstracts? I wish I had easy access to them)
1. Use Mangaki's [zero](https://github.com/mangaki/zero) algorithm for recommendation (or just logistic regression)
1. Make it a simple service (before the conference ends)

Here it is!

    make
    python adaptive.py

Then just answer the questions and you're all set. Have a nice conference!
