from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
import re
import operator
import matplotlib
from gensim.models.keyedvectors import KeyedVectors

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import OrderedDict

# download example data ( may take a while)
#train = fetch_20newsgroups()

file = "data/embeddings/bible_word2vec"


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    words = 300
    i = 0

    #sorted_vocabs  = sorted(model.vocab.items(), key= operator.attrgetter('count'), reverse=True)


    for word in model.vocab:
        #print("here")
        tokens.append(model[word])
        labels.append(word)
        if i > words:
            break
        else:
            i = i + 1

    tsne_model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=5000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 12))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    print("here")
    plt.show()


model = KeyedVectors.load(file)
#model = OrderedDict(model)
tsne_plot(model)