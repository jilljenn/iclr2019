from gensim.models.fasttext import load_facebook_vectors
# from gensim.models.wrappers import FastText


meow = load_facebook_vectors('wiki.simple.bin')
print(meow)

model = FastText.load_fasttext_format('wiki.simple')

print(model.most_similar('teacher'))
# Output = [('headteacher', 0.8075869083404541),
#           ('schoolteacher', 0.7955552339553833),
#           ('teachers', 0.733420729637146), ('teaches', 0.6839243173599243),
#           ('meacher', 0.6825737357139587), ('teach', 0.6285147070884705),
#           ('taught', 0.6244685649871826), ('teaching', 0.6199781894683838),
#           ('schoolmaster', 0.6037642955780029),
#           ('lessons', 0.5812176465988159)]

print(model.similarity('teacher', 'teaches'))
# Output = 0.683924396754
