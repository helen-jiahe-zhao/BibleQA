from deep_qa import run_model, evaluate_model, load_model, score_dataset
from deep_qa.layers import highway, complex_concat
from deep_qa.layers.encoders import seq2seq_encoders
from keras.layers.wrappers import Bidirectional

from deep_qa.layers.attention import MatrixAttention, MaskedSoftmax, WeightedSum
from deep_qa.layers.backend import Max, RepeatLike, Repeat

from deep_qa.training import TextTrainer
from deep_qa.training.models import DeepQaModel
from deep_qa.common.params import Params

from deep_qa.models.reading_comprehension import bidirectional_attention
from deep_qa.training import text_trainer
from deep_qa.common import params

from keras.layers.wrappers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from keras import layers
from keras import models
from keras import backend as K
from keras import metrics
from keras import losses
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.regularizers import l2 # L2-regularisation

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from nltk.corpus import stopwords

import csv
import os
import re
import numpy as np
import json
import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.corpora import WikiCorpus

from gensim.models import word2vec



"""
SOME GLOBAL VARIABLES
"""
np.random.seed(212)

BIBLEQA_DIR = "data/bible_qa/"
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
    dir = BIBLEQA_DIR + "bible_qa_no_context.csv"

    #in list formats
    questions = []
    answers = []
    labels = []
    """
    with open(dir, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter = "\t")
        next(reader)
        #'ID', 'Question', 'Verse_Code', 'KJV_Verse', 'ASV_Verse', 'YLT_Verse', 'WEB_Verse', 'Label'
        for id, question, verse_code, kjv, asv, ylt, web, label in reader:
            all_questions.append(question)
            all_answers.append(kjv)
            y.append(int(label))

            all_questions.append(question)
            all_answers.append(asv)
            y.append(int(label))

            all_questions.append(question)
            all_answers.append(ylt)
            y.append(int(label))

            all_questions.append(question)
            all_answers.append(web)
            y.append(int(label))
    """

    with open('data/bible_qa/bible_qa_list_3_asv.json', 'r') as f:
        data = json.load(f)
        for item in data:
            for version_answer in item["answers"]:

                questions.append(item["question"])
                answers.append(version_answer)
                labels.append(item["labels"])


    return {"Q": questions, "A": answers, "L": labels}


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

    print("combined len ", len(combined_index.keys()))
    print("combined dim ", len(combined_index["love"]))

    return combined_index

def get_embeddings():
    # build index mapping words in the embeddings set
    # to their embedding vector

    #glove vectors
    glove_embeddings_index = {}
    f = open(os.path.join(EMBEDDING_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_embeddings_index[word] = coefs
    f.close()

    #word2vec model
    bible_embeddings_index = {}
    bible_model = KeyedVectors.load(EMBEDDING_DIR + 'bible_word2vec')
    bible_vocab = bible_model.vocab

    for word in bible_vocab:
        bible_embeddings_index[word] = bible_model[word]
        #print("WORD: ", word)
        #print("VECTOR: ", bible_model[word])


    combined_index = combine_embeddings(glove_embeddings_index, bible_embeddings_index)

    print('Found word vectors: ', len(combined_index))
    print('dimensions: ', len(combined_index["god"]))
    print('dimensions: ', len(combined_index["bottle"]))

    return combined_index

def vectorise_and_split_texts(data):

    # finally, vectorize the text samples into a 2D integer tensor
    all_questions = data["Q"]
    all_answers = data["A"]
    labels = data["L"]

    all_texts = []

    for question_list in all_questions:
        for question in question_list:
            all_texts = all_texts + [clean_sentence(question)]
            break

    for answer_list in all_answers:
        for answer in answer_list:
            all_texts = all_texts + [clean_sentence(answer)]



    tokenizer.fit_on_texts(all_texts)

    #print("quesiton sequence before ", all_questions[:10])
    #print("answer_sequences before", all_answers[:10])


    """split into train and test"""


    num_validation_samples = int(VALIDATION_SPLIT * len(labels))

    question_train = all_questions[:-num_validation_samples]
    answer_train = all_answers[:-num_validation_samples]
    labels_train = labels[:-num_validation_samples]

    question_test = all_questions[-num_validation_samples:]
    answer_test = all_answers[-num_validation_samples:]
    labels_test = labels[-num_validation_samples:]

    """deal with train"""
    q = []
    a = []
    l = []
    for i in range(0, len(question_train)):
        q = q + question_train[i]
        a = a + answer_train[i]
        l = l + labels_train[i]
    question_train = q
    answer_train = a
    labels_train = l

    question_sequences = tokenizer.texts_to_sequences(question_train)
    answer_sequences = tokenizer.texts_to_sequences(answer_train)


    train_question_data = pad_sequences(question_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    train_answer_data = pad_sequences(answer_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    train_label_data = np.asarray(labels_train)

    """deal with test"""

    test_question_data = []
    test_answer_data = []
    test_label_data = []#np.asarray(labels_test)

    for i in range(0, len(question_test)):
        question_sequences = tokenizer.texts_to_sequences(question_test[i])
        answer_sequences = tokenizer.texts_to_sequences(answer_test[i])

        question_data = pad_sequences(question_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        answer_data = pad_sequences(answer_sequences, maxlen=MAX_SEQUENCE_LENGTH)

        test_question_data.append(question_data)
        test_answer_data.append(answer_data)
        test_label_data = test_label_data + labels_test[i]

    test_label_data = np.asarray(test_label_data)



    return {"train": [train_question_data, train_answer_data, train_label_data], "test": [test_question_data, test_answer_data, test_label_data]}


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
                                trainable=False, name = "bible")
    return embedding_layer


def build_model(embedding_layer): #, params: Params):

    question_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,),
                                  dtype='int32')  # * 2 since doubling the question and passage
    answer_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,),
                                dtype='int32')  # * 2 since doubling the question and passage

    question_embedding = embedding_layer(question_input)
    answer_embedding = embedding_layer(answer_input)

    # Min's model has some highway layers here, with relu activations.  Note that highway
    # layers don't change the tensor's shape.  We need to have two different `TimeDistributed`
    # layers instantiated here, because Keras doesn't like it if a single `TimeDistributed`
    # layer gets applied to two inputs with different numbers of time steps.
    highway_layers = 2
    for i in range(highway_layers):
        highway_layer = highway.Highway(activation='relu', name='highway_{}'.format(i))
        question_layer = layers.TimeDistributed(highway_layer, name=highway_layer.name + "_qtd", trainable=False)
        question_embedding = question_layer(question_embedding)
        passage_layer = layers.TimeDistributed(highway_layer, name=highway_layer.name + "_ptd", trainable=False)
        answer_embedding = passage_layer(answer_embedding)

    # Then we pass the question and passage through a seq2seq encoder (like a biLSTM).  This
    # essentially pushes phrase-level information into the embeddings of each word.
    phrase_layer = Bidirectional(layers.GRU(return_sequences=True, units=500, activation='relu', recurrent_dropout= 0.2, dropout=0.3, trainable=False ))#, **(params["encoder_params"]), **(params["wrapper_params"])))

    # Shape: (batch_size, num_question_words, embedding_dim * 2)
    encoded_question = phrase_layer(question_embedding)

    # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    encoded_answer = phrase_layer(answer_embedding)

    # PART 2:
    # Now we compute a similarity between the passage words and the question words, and
    # normalize the matrix in a couple of different ways for input into some more layers.
    matrix_attention_layer = MatrixAttention(similarity_function={'type': 'linear', 'combination': 'x,y,x*y'},
                                             name='passage_question_similarity', trainable=False)

    # Shape: (batch_size, num_passage_words, num_question_words)
    answer_question_similarity = matrix_attention_layer([encoded_answer, encoded_question])

    # Shape: (batch_size, num_passage_words, num_question_words), normalized over question
    # words for each passage word.
    answer_question_attention = MaskedSoftmax()(answer_question_similarity)
    # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    weighted_sum_layer = WeightedSum(name="answer_question_vectors", use_masking=False, trainable=False)
    answer_question_vectors = weighted_sum_layer([encoded_question, answer_question_attention])

    # Min's paper finds, for each document word, the most similar question word to it, and
    # computes a single attention over the whole document using these max similarities.
    # Shape: (batch_size, num_passage_words)
    question_answer_similarity = Max(axis=-1)(answer_question_similarity)
    # Shape: (batch_size, num_passage_words)
    question_answer_attention = MaskedSoftmax()(question_answer_similarity)
    # Shape: (batch_size, embedding_dim * 2)
    weighted_sum_layer = WeightedSum(name="question_passage_vector", use_masking=False, trainable=False)
    question_answer_vector = weighted_sum_layer([encoded_answer, question_answer_attention])

    # Then he repeats this question/passage vector for every word in the passage, and uses it
    # as an additional input to the hidden layers above.
    repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
    # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    tiled_question_answer_vector = repeat_layer([question_answer_vector, encoded_answer])

    # Shape: (batch_size, num_passage_words, embedding_dim * 8)
    complex_concat_layer = complex_concat.ComplexConcat(combination='1,2,1*2,1*3', name='final_merged_passage')
    final_merged_answer = complex_concat_layer([encoded_answer,
                                                 answer_question_vectors,
                                                 tiled_question_answer_vector])

    # PART 3:
    # Having computed a combined representation of the document that includes attended question
    # vectors, we'll pass this through a few more bi-directional encoder layers, then predict
    # the span_begin word.  Hard to find a good name for this; Min calls this part of the
    # network the "modeling layer", so we'll call this the `modeled_passage`.
    modeled_answer = final_merged_answer
    for i in range(1):
        hidden_layer = Bidirectional(layers.GRU(return_sequences=True, units=300, activation='relu', recurrent_dropout= 0.2, dropout=0.3, ))#, **(params["encoder_params"]), **(params["wrapper_params"])))
        modeled_answer = hidden_layer(modeled_answer)


    #PART 4: BY HELEN
    #get the maximum for each word
    max_answer = Max(axis=-1)(modeled_answer)
    print("max answer shape", max_answer.shape)
    print("modeled_answer shape", modeled_answer.shape)


    preds = layers.Dense(1, activation = 'sigmoid', name = 'prediction')(max_answer)

    print("pred shape", preds.shape)

    model = models.Model(inputs=[question_input, answer_input], outputs=preds)

    return model

"""
custom metrics
"""

def create_and_train_model(splitted_data, embedding_layer):
    train = splitted_data["train"]
    test = splitted_data["test"]
    question_train = train[0]
    answer_train = train[1]
    label_train = train[2]

    question_test = test[0]
    answer_test = test[1]
    label_test = test[2]



    #
    #learning_rate_list = [0.01, 0.001, 0.0001, 0.00001]

    #for lr in learning_rate_list:
    model = build_model(embedding_layer)
    # model = bidirectional_attention.BidirectionalAttentionFlow(params.Params).model()

    model.load_weights("bidaf_squad_model.h5", by_name=True)

    model.compile(loss='binary_crossentropy', #loss = 'categorical_crossentropy', #
                  #optimizer=optimizers.adagrad(lr=0.01, epsilon=1e-08),
                  optimizer=optimizers.adagrad(lr=0.001),
                  metrics=['accuracy'])
                  #metrics=[precision_threshold(0.5), recall_threshold(0.5)])
                  #class_mode='binary')

    model.fit([question_train, answer_train], label_train,
              batch_size=128,
              epochs=30,
              validation_split = 0.1, verbose = 1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
              #show_accuracy=True
              )
              #validation_data=([question_test, answer_test], label_test))

    label_predicted = []
    divide_list = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
    #divided_label_test = divide_list(label_test,10)
    reciprocal_ranks = []


    label_index_begin = 0
    list_size = 0
    for i in range(0, len(question_test)):
        q = question_test[i]
        a = answer_test[i]
        current_predicted = model.predict([q, a])
        binary_predicted = [0] * len(current_predicted)
        highest_index = np.argmax(current_predicted)
        #highest = current_predicted[i]
        #for num in range(0, len(current_predicted)):
        #    if current_predicted[num] > highest:
        #        highest = current_predicted[num]
        #        highest_index = num
        binary_predicted[highest_index] = 1
        label_predicted = label_predicted + binary_predicted

        #get the rank here
        current_answer_list_length = len(a)
        label_index_end = label_index_begin + current_answer_list_length
        current_label_list = label_test[label_index_begin:label_index_end]

        index_of_1 = np.where(current_label_list == 1)[0][0] #e.g. 2
        #value_of_the_item_thats_supposed_to_be_highest = current_predicted[index_of_1]
        sorted_by_index = np.argsort(current_predicted.flatten()) #e.g. [0,2,1]
        actual_rank = np.where(sorted_by_index == index_of_1)[0][0] + 1

        reciprocal_ranks = reciprocal_ranks + [1/actual_rank]
        #print("recip rank", reciprocal_ranks[i])
        list_size = list_size + 1

    input_size = len(question_test)
    #reciprocal_ranks = np.array(reciprocal_ranks)
    sum_of_ranks = np.sum(reciprocal_ranks)
    mrr = 1 / input_size * sum_of_ranks

    label_predicted = np.asarray(label_predicted)
    #label_predicted = model.predict([question_test, answer_test])
    print("predicted: " , label_predicted[:100])
    print("actual: " , label_test[:100])

    precision = precision_score(label_test, label_predicted)
    recall = recall_score(label_test, label_predicted)
    f1 = f1_score(label_test, label_predicted)

    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    print("mrr: ", mrr)

if __name__ == '__main__':

    print("--- GETTING DATASET ---")
    data = read_data()

    print("--- VECTORISING TEXT SAMPLES --- ")
    vectorised_data = vectorise_and_split_texts(data)

    print("--- SPLITTING DATA ---")
  #  splitted_data = split_data(vectorised_data)

    print("--- INDEXING WORD VECTORS ---")
    embeddings_index = get_embeddings()

    print("--- PREPARING EMBEDDING LAYER ---")
    embedding_layer = prepare_embedding_mat(embeddings_index)

    print("--- MODELLING --- ")
    create_and_train_model(vectorised_data, embedding_layer)


