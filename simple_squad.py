from deep_qa import run_model, evaluate_model, load_model, score_dataset
from deep_qa.layers import *
from deep_qa.layers.backend import *
from deep_qa.layers.attention import *

from deep_qa.models.reading_comprehension import bidirectional_attention
from deep_qa.training import text_trainer
from deep_qa.common import params

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from keras import layers
from keras import models
from keras.layers import Dense, Input, Concatenate, TimeDistributed
from keras.constraints import maxnorm

from keras import backend as K
from keras import metrics
from keras import losses
from keras import optimizers
from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from nltk.corpus import stopwords

import csv
import os
import re
import json
import numpy as np
import nltk
import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.corpora import WikiCorpus

from gensim.models import word2vec



"""
SOME GLOBAL VARIABLES
"""
np.random.seed(212)


SQUAD_DIR = "data/squad/"
EMBEDDING_DIR = 'data/embeddings/'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 100
TOTAL_EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.3

tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False)

"""
1. read data
"""

def read_data():
    train_dir = SQUAD_DIR + "train-v1.1-sentence.json"
    dev_dir = SQUAD_DIR + "dev-v1.1-sentence.json"


    train_questions = []
    train_answers = []
    train_y = []

    dev_questions = []
    dev_answers = []
    dev_y = []

    with open(train_dir, 'r') as f:
        train_data = json.load(f)['data']
        for item in train_data:
            train_questions.append(clean_sentence(item[1]))
            train_answers.append(clean_sentence(item[2]))
            train_y.append(item[3])

    with open(dev_dir, 'r') as f:
        dev_data = json.load(f)['data']
        for item in dev_data:
            dev_questions.append(clean_sentence(item[1]))
            dev_answers.append(clean_sentence(item[2]))
            dev_y.append(item[3])

    return {"train":[train_questions, train_answers, train_y], "dev":[train_questions, train_answers, train_y]}

def clean_sentence(text):
    #print("before: ", text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    # changing contractions
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'s", "", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)

    text = re.sub(' +', ' ', text)
    text = text.split()

    text = [w.lower() for w in text]

    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]

    text = " ".join(text)

    #print("after: ", text)

    return text


def combine_embeddings(glove, bible):
    print("glove len ", len(glove))
    print("bible len ", len(bible))
    print("glove dim ", len(glove["bottle"]))
    print("bible dim ", len(bible["god"]))

    glove_dim = len(glove["bottle"])
    bible_dim = len(bible["god"])

    new_dim = glove_dim + bible_dim

    empty_glove_vector = np.zeros(shape=(glove_dim,))
    empty_bible_vector = np.zeros(shape=(bible_dim,))

    glove_vocab = glove.keys()
    bible_vocab = bible.keys()
    combined_vocab = list(glove_vocab) + list(bible_vocab)

    combined_index = {}

    for word in combined_vocab:
        glove_vector = glove[word] if word in glove_vocab else empty_glove_vector
        bible_vector = bible[word] if word in bible_vocab else empty_bible_vector
        combined_vector = np.concatenate([glove_vector,bible_vector])
        combined_index[word] = combined_vector

    #print("combined len ", len(combined_index.keys()))
    #print("combined dim ", len(combined_index["love"]))

    return combined_index

def get_embeddings():
    # build index mapping words in the embeddings set
    # to their embedding vector

    #glove vectors
    glove_embeddings_index = {}
    f = open(os.path.join(EMBEDDING_DIR, 'glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_embeddings_index[word] = coefs
    f.close()
    """
    #word2vec model
    bible_embeddings_index = {}
    bible_model = KeyedVectors.load(EMBEDDING_DIR + 'bible_word2vec')
    bible_vocab = bible_model.vocab

    for word in bible_vocab:
        bible_embeddings_index[word] = bible_model[word]
        #print("WORD: ", word)
        #print("VECTOR: ", bible_model[word])


    combined_index = combine_embeddings(glove_embeddings_index, bible_embeddings_index)

    #print('Found word vectors: ', len(combined_index))
    #print('dimensions: ', len(combined_index["god"]))
    #print('dimensions: ', len(combined_index["bottle"]))
    """
    return glove_embeddings_index

def vectorise_texts(data):
    # finally, vectorize the text samples into a 2D integer tensor
    #all_questions = data["Q"]
    #all_answers = data["A"]
    #labels = data["L"]

    train = data['train']
    dev = data['dev']
    #print(labels[1:100])
    all_texts = train[0] + train[1] + dev[0] + dev[1]


    """
    cleaned_questions = []
    for question in all_questions:
        cleaned_questions.append(clean_sentence(question))
    all_questions = cleaned_questions


    cleaned_answers = []
    for answer in all_answers:
        cleaned_answers.append(clean_sentence(answer))
        all_answers = cleaned_answers
    """
    tokenizer.fit_on_texts(all_texts)

    #print("quesiton sequence before ", train[0][:10])
    #print("answer_sequences before", train[1][:10])

    train_question_sequences = tokenizer.texts_to_sequences(train[0])
    train_answer_sequences = tokenizer.texts_to_sequences(train[1])

    dev_question_sequences = tokenizer.texts_to_sequences(dev[0])
    dev_answer_sequences = tokenizer.texts_to_sequences(dev[1])

    #print("quesiton sequence", train_question_sequences[:10])
    #print("answer_sequences", train_answer_sequences[:10])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    train_question_data = pad_sequences(train_question_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    train_answer_data = pad_sequences(train_answer_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    dev_question_data = pad_sequences(dev_question_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    dev_answer_data = pad_sequences(dev_answer_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    #print("words: ", word_index)

    #print("padded quesiton sequence", train_question_data[:10])
    #print("padded answer_sequences", train_answer_data[:10])

    train_labels = np.asarray(train[2])
    dev_labels = np.asarray(dev[2])

    #labels = utils.to_categorical(np.asarray(labels))
    print('Shape of question data tensor:', train_question_data.shape)
    print('Shape of answer data tensor:', train_answer_data.shape)
    print('Shape of dev label tensor:', dev_labels.shape)
    #print('labels:' , labels[1:100])

    return {"train":[train_question_data, train_answer_data, train_labels], "dev":[dev_question_data, dev_answer_data, dev_labels]}

"""
def split_data(data):
    # split the data into a training set and a validation set

    question_data = data["Q"]
    answer_data = data["A"]
    labels = data["L"]

    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    #print('indices', indices[0:100])

    question_data = question_data[indices]
    answer_data = answer_data[indices]
    labels = labels[indices]

    num_validation_samples = int(VALIDATION_SPLIT * labels.shape[0])

    question_train = question_data[:-num_validation_samples]
    answer_train = answer_data[:-num_validation_samples]
    labels_train = labels[:-num_validation_samples]

    question_test = question_data[-num_validation_samples:]
    answer_test = answer_data[-num_validation_samples:]
    labels_test = labels[-num_validation_samples:]


    return {"QT": question_train, "AT": answer_train, "LT": labels_train, "QTest": question_test, "ATest": answer_test, "LTest": labels_test}
"""
def prepare_embedding_mat(embeddings_index):
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS+1, len(tokenizer.word_index)+1)
    embedding_matrix = np.zeros((num_words, TOTAL_EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i == 0 :
            print("has a 0")
        #print(word)
        #print(i)
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            #-1 because the index has i starting from 1
            embedding_matrix[i] = embedding_vector

    #print("embeddingMatrix shape:", embedding_matrix.shape)
    #print(embedding_matrix[0:100])

    embedding_layer = layers.Embedding(num_words,
                                TOTAL_EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False, name = "squad_embed")
    return embedding_layer


def build_model(embedding_layer):

    #model = bidirectional_attention.BidirectionalAttentionFlow(params=params.Params())

    # train a 1D convnet with global maxpooling
    question_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,),
                                  dtype='int32')  # * 2 since doubling the question and passage
    answer_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,),
                                dtype='int32')  # * 2 since doubling the question and passage

    question_embedding = embedding_layer(question_input)
    answer_embedding = embedding_layer(answer_input)

    question = layers.LSTM(units=300, activation='relu', recurrent_dropout= 0.2, dropout=0.3, name = "question")(question_embedding)
    answer = layers.LSTM(units=300, activation='relu', recurrent_dropout= 0.2, dropout=0.3, name = "answer")(answer_embedding)
    #question = layers.Dense(100, activation = 'relu')(question)
    #answer = layers.Dense(100, activation = 'relu')(answer)
    #question = layers.Dropout(0.3)(question)
    #answer = layers.Dropout(0.3)(answer)

    #preds = layers.dot([question, answer], normalize=True, axes=-1)

    merged = layers.concatenate([question, answer])
    preds = layers.Dense(1, activation='sigmoid', name="activation")(merged)  # len(label_train)

    #merged = layers.Dropout(0.3)(merged)
    #merged = layers.Dense(200, activation='relu', name = "dense1")(merged)
    #merged = layers.Dropout(0.3)(merged)
    #merged = layers.Dense(50, activation='relu', name = "dense2")(merged)#, kernel_constraint=maxnorm(4)
    #merged = layers.Dropout(0.3)(merged)

    #preds = layers.Dense(1, activation='sigmoid', name = "activation")(merged)  # len(label_train)

    model = models.Model(inputs=[question_input, answer_input], outputs=preds)

    return model

def create_and_train_model(splitted_data, embedding_layer):
    question_train = splitted_data["train"][0]
    answer_train = splitted_data["train"][1]
    label_train = splitted_data["train"][2]

    question_test = splitted_data["dev"][0]
    answer_test = splitted_data["dev"][1]
    label_test = splitted_data["dev"][2]

    model = build_model(embedding_layer)

    model.compile(loss='binary_crossentropy', #loss = 'categorical_crossentropy', #
                  #optimizer=optimizers.adagrad(lr=0.01, epsilon=1e-08),
                  optimizer=optimizers.adagrad(lr=0.0001),
                  metrics=['accuracy'])
                  #metrics=[precision_threshold(0.5), recall_threshold(0.5)])
                  #class_mode='binary')

    model.fit([question_train, answer_train], label_train,
              batch_size=128,
              epochs=30,
              validation_split = 0.1, verbose = 2
              #show_accuracy=True
              )
              #validation_data=([question_test, answer_test], label_test))
    #model_json = model.to_json()
    #with open("squad_model.json", "w") as json_file:
    #    json_file.write(model_json)
    model.save_weights("simple_squad_model_2.h5")

    label_predicted = model.predict([question_test, answer_test])
    print("predicted: " , label_predicted[:100])
    print("actual: " , label_test[:100])

    score = model.evaluate([question_test, answer_test], label_test, verbose=1)

    print("keras score: ", score)

    label_binary = []
    for list in label_predicted:
        for num in list:
            threshold = 0.5
            if num > threshold:
                label_binary.append(1)
            else:
                label_binary.append(0)

    label_binary = np.asarray(label_binary)
    print(label_binary[:100])
    precision = precision_score(label_test, label_binary)
    recall = recall_score(label_test, label_binary)
    f1 = f1_score(label_test, label_binary)

    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)

if __name__ == '__main__':

    print("--- GETTING DATASET ---")
    data = read_data()

    print("--- VECTORISING TEXT SAMPLES --- ")
    vectorised_data = vectorise_texts(data)

    print("--- SPLITTING DATA ---")
    #splitted_data = split_data(vectorised_data)

    print("--- INDEXING WORD VECTORS ---")
    embeddings_index = get_embeddings()

    print("--- PREPARING EMBEDDING LAYER ---")
    embedding_layer = prepare_embedding_mat(embeddings_index)

    print("--- MODELLING --- ")
    create_and_train_model(vectorised_data, embedding_layer)


