import csv
import re
import logging

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


from string import punctuation
from gensim.models import word2vec
#from gensim.models import keyedvectors
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import re
import matplotlib
from gensim.models.keyedvectors import KeyedVectors
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
SETTING FOR PREPROCESSING
"""
remove_stopwords = True
stem_words = False
lemmatisation = False


def get_tag(treebank_tag):
    treebank_tag = treebank_tag[0][1]
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'a'
    else:
        return 'n'

def text_to_word_list(text):
    #if '{' or '}' in text:

    #print("BEFORE: ", text)

    text = text.lower()
    #taking out anything that is not letters or numbers
    text = re.sub(r"[^A-Za-z0-9!:?]", " ", text)
    #changing contractions
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'s", "", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)

    #keep punctuations for now
    text = re.sub(r",", "", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"!", " !", text)
    text = re.sub(r"\?", " ?", text)

    text = re.sub(r"\/", "", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r":", " :", text)

    #remove extra white space
    text = re.sub(' +', ' ', text)

    text = text.split()

    #optionally remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        #print("SW REMOVED ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        stemmer = SnowballStemmer('english')
        text = [stemmer.stem(word) for word in text]
        #print("STEMMED ", text)
        #text = " ".join(stemmed_words)

    if lemmatisation:
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word, get_tag(nltk.pos_tag([word]))) for word in text]
        #print("LEMMATISED ", text)
    #print("AFTER: ", text)

    return text

def import_bible():
    bible = []

    kjv_dir = "../data/bible/t_kjv.csv"
    asv_dir = "../data/bible/t_asv.csv"
    ylt_dir = "../data/bible/t_ylt.csv"
    wbt_dir = "../data/bible/t_wbt.csv"


    count = 1
    with open(kjv_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, book, chap, verse, text in reader:
            if count % 1000 == 0:
                print ("PROCESSING UP TO LINE ", count)
            #strip weird characters
            #and split the verse into a list of words
            bible.append(text_to_word_list(text))
            count = count + 1


    with open(asv_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, book, chap, verse, text in reader:
            if count % 1000 == 0:
                print("PROCESSING UP TO LINE ", count)
            #strip weird characters
            #and split the verse into a list of words
            bible.append(text_to_word_list(text))
            count = count + 1



    with open(ylt_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, book, chap, verse, text in reader:
            if count % 1000 == 0:
                print("PROCESSING UP TO LINE ", count)
            #strip weird characters
            #and split the verse into a list of words
            bible.append(text_to_word_list(text))
            count = count + 1


    with open(wbt_dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for id, book, chap, verse, text in reader:
            if count % 1000 == 0:
                print("PROCESSING UP TO LINE ", count)
            #strip weird characters
            #and split the verse into a list of words
            bible.append(text_to_word_list(text))
            count = count + 1

    print("LINES IN BIBLE: ", len(bible))
    print(bible[0])
    print(bible[1000])

    return(bible)


def create_w2v_model(bible):
    min_count = 1500 #words that appear at least this many times is included
    size = 200 #how many layers
    window = 5 #sliding window size
    workers = 4 #parallelism

    model = word2vec.Word2Vec(bible, min_count=min_count, size=size, window=window, workers=workers)
    word_vectors = model.wv
    print(word_vectors.most_similar(positive=['god']))
    return word_vectors

def visualise(model):
    labels = []
    tokens = []

    for word in model.vocab:
        tokens.append(model[word])
        labels.append(word)

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
    plt.show()


if __name__ == '__main__':
    bible = import_bible()
    model = create_w2v_model(bible)
    visualise(model)


    #model.save("data/embeddings/bible_word2vec_no_lemma")

