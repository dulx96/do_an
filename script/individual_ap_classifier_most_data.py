import pandas as pd
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Input, concatenate, Embedding, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.preprocessing.sequence import pad_sequences
import nltk
import re
import string

import helpers

from keras import backend as K

import os


# custom evaluation function
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


#
def create_tokenizer(text_array):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_array)
    return tokenizer


#
def get_pretrained_embedding(file, tokenizer, vocab_size, dimensions):
    """generate embedding matrix from source file"""
    embedding_index = dict()
    f = open(file, mode='rt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((vocab_size, dimensions))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# load most common word
def load_most_common_word(file, ap_list):
    vocab = set()
    for ap in ap_list:
        path = file + '/' + ap + '.txt'
        vocab.update(helpers.load_doc(path).split())
    return vocab


# get all input, output X,Y
def prepare_X_dict(data_train, vocab, vocab_most_common):
    """
    :param data_train - pdframe
    :param data_test - pdframe
    :param vocab set of words
    generate all info  about each input data, train and test"""
    x_dict = list()
    # X1,review input,w2v
    X1_train_texts = X1_process_texts(data_train.text, vocab)
    X1_tokenizer = create_tokenizer(X1_train_texts)
    X1_vocab_size = len(X1_tokenizer.word_index) + 1
    X1_max_length = max([len(s.split()) for s in X1_train_texts])
    X1_embedding_matrix = get_pretrained_embedding(embedding_file, X1_tokenizer, X1_vocab_size, 100)

    def X1_transform_text_array(text_array):
        X1_data = X1_process_texts(text_array, vocab)
        X1_data = X1_encode(X1_data, X1_tokenizer, X1_max_length)
        return X1_data

    X1_dict = {"max_length": X1_max_length, "embedding_matrix": X1_embedding_matrix, "vocab_size": X1_vocab_size,
               "transform_function": X1_transform_text_array}
    x_dict.append(X1_dict)

    # X2,review input,w2v
    X2_train_texts = X1_process_texts(data_train.text, vocab)
    X2_tokenizer = create_tokenizer(X2_train_texts)
    X2_vocab_size = len(X2_tokenizer.word_index) + 1
    X2_max_length = max([len(s.split()) for s in X2_train_texts])
    X2_embedding_matrix = get_pretrained_embedding(res_embedding_file, X2_tokenizer, X2_vocab_size, 100)

    def X2_transform_text_array(text_array):
        X_data = X1_process_texts(text_array, vocab)
        X_data = X1_encode(X_data, X2_tokenizer, X2_max_length)
        return X_data

    X2_dict = {"max_length": X2_max_length, "embedding_matrix": X2_embedding_matrix, "vocab_size": X2_vocab_size,
               "transform_function": X2_transform_text_array}
    x_dict.append(X2_dict)

    # X3, Noun, bag of word
    X3_train_texts = X3_process_texts(data_train.text, vocab)
    X3_tokenizer = create_tokenizer(X3_train_texts)
    X3_max_length = len(X3_tokenizer.word_index) + 1

    def X3_transform_text_array(text_array):
        X_data = X3_process_texts(text_array, vocab)
        X_data = X3_encode(X_data, X3_tokenizer)
        return X_data

    X3_dict = {"max_length": X3_max_length, "transform_function": X3_transform_text_array}
    x_dict.append(X3_dict)

    # X4, most comoon Noun adj
    X4_tokenizer = create_tokenizer(vocab_most_common)
    X4_max_length = len(X4_tokenizer.word_index) + 1

    def X4_transform_text_array(text_array):
        X_data = X4_process_texts(text_array, vocab)
        X_data = X4_encode(X_data, X4_tokenizer)
        return X_data

    X4_dict = {"max_length": X4_max_length, "transform_function": X4_transform_text_array}
    x_dict.append(X4_dict)

    return x_dict


# define model
def define_model(x_dict_list):
    """
    :param data - list of dict X
    gen model from input data info"""
    # X1
    X1 = x_dict_list[0]
    X1_max_length = X1["max_length"]
    X1_vocab_size = X1["vocab_size"]
    X1_embedding_matrix = X1["embedding_matrix"]
    X1_input = Input(shape=(X1_max_length,))
    X1_embedding = Embedding(X1_vocab_size, 100, weights=[X1_embedding_matrix])(X1_input)
    X1_conv = Conv1D(filters=100, kernel_size=2, activation='relu')(X1_embedding)
    X1_drop = Dropout(0.1)(X1_conv)
    X1_pool = MaxPooling1D(pool_size=2)(X1_drop)
    X1_flat = Flatten()(X1_pool)
    X1_output = X1_flat

    # X2
    X2 = x_dict_list[1]
    X2_max_length = X2["max_length"]
    X2_vocab_size = X2["vocab_size"]
    X2_embedding_matrix = X2["embedding_matrix"]
    X2_input = Input(shape=(X2_max_length,))
    X2_embedding = Embedding(X2_vocab_size, 100, weights=[X2_embedding_matrix])(X2_input)
    X2_conv = Conv1D(filters=300, kernel_size=2, activation='relu')(X2_embedding)
    X2_drop = Dropout(0.1)(X2_conv)
    X2_pool = MaxPooling1D(pool_size=2)(X2_drop)
    X2_flat = Flatten()(X2_pool)
    X2_output = X2_flat
    # X3
    X3 = x_dict_list[2]
    X3_max_length = X3["max_length"]
    X3_input = Input(shape=(X3_max_length,))
    X3_output = X3_input

    # X4
    X4 = x_dict_list[3]
    X4_max_length = X4["max_length"]
    X4_input = Input(shape=(X4_max_length,))
    X4_output = X4_input

    # model
    merged = concatenate([X1_output, X2_output, X3_output, X4_output])
    # dense1 = Dense(512, activation='relu')(merged)
    dense2 = Dense(100, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense2)
    model = Model(inputs=[X1_input, X2_input, X3_input, X4_input], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
    return model


def filter_data_with_ap(ap, data):
    ap = ap.split('#')[0]
    temp_csv = data[data.aspect_category.str.contains(ap)].reset_index()
    return temp_csv


def train(x_dict_list, data_train, data_test):
    for ap in aspect_category_list:
        print(ap)
        model = define_model(x_dict_list)
        model.summary()
        plot_model(model, show_shapes=True, to_file=model_folder + '/' + model_file_name + '/' + ap + '.png')
        Y1_train = Y1_encode(ap, data_train)
        X_train = [X["transform_function"](data_train.text) for X in x_dict_list]
        model.fit(X_train, Y1_train, epochs=100, verbose=2)
        evaluate_model(model, ap, x_dict_list, data_test)
        model.save(model_folder + '/' + model_file_name + '/' + ap + 'model.h5')


def load_model_list():
    model = []
    for ap in aspect_category_list:
        model.append(
            {'aspect_category': ap,
             'model': load_model(model_folder + '/' + model_file_name + '/' + ap + 'model.h5',
                                 custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})}
        )
    return model


def evaluate_model(model, aspect_category, x_dict_list, data_test):
    Y1_test = Y1_encode(aspect_category, data_test)
    X_test_array = [X["transform_function"](data_test.text) for X in x_dict_list]
    _, acc, f1_score, precision, recall = model.evaluate(X_test_array, Y1_test, verbose=2)
    print('%s Accuracy: %f' % (aspect_category, acc * 100))
    print('%s F1_score: %f' % (aspect_category, f1_score * 100))
    print('%s Precision: %f' % (aspect_category, precision * 100))
    print('%s Recall: %f' % (aspect_category, recall * 100))


def evaluate_model_list(model_list, x_dict_list, data_test):
    for model in model_list:
        evaluate_model(model['model'], model['aspect_category'], x_dict_list, data_test)


def predict_outside(text_array):
    text_predict = [X["transform_function"](text_array) for X in X_dict_list]
    # print(text_array)
    # print(text_predict)
    predicted = []
    for model in model_list:
        y_hat = model["model"].predict(text_predict)
        predicted.append({"ap": model["aspect_category"], "H": y_hat[0, 0] * 100})
    return predicted


# X1
# word embedding 100 dimension from glove
def X1_clean_text(text, vocab):
    tokens = helpers.clean_text_to_tokens_3(text)
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    texts = ' '.join(tokens)
    return texts


def X1_process_texts(text_array, vocab):
    texts_clean_list = list()
    for text in text_array:
        texts_clean_list.append(X1_clean_text(text, vocab))
    return texts_clean_list


def X1_encode(text_array, tokenizer, max_length):
    encoded = tokenizer.texts_to_sequences(text_array)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


# X3, all Noun from text

X3_clean_text = X1_clean_text


def X3_process_texts(text_array, vocab):
    texts_clean_list = list()
    is_noun = lambda pos: pos[:2] == 'NN'
    is_adj = lambda pos: pos[:2] == 'JJ'
    for text in text_array:
        text = X3_clean_text(text, vocab)
        tokenized = text.split()
        filters = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos) or is_adj(pos)]
        texts_clean_list.append(' '.join(filters))
    return texts_clean_list


def X3_encode(text_array, tokenizer):
    encoded = tokenizer.texts_to_matrix(text_array)
    return encoded


# X4 most common Noun, adj
X4_clean_text = X1_clean_text
X4_process_texts = X1_process_texts


def X4_encode(text_array, tokenizer):
    encoded = tokenizer.texts_to_matrix(text_array)
    return encoded


def Y1_encode(aspect_category, data):
    """
    :param aspect_category
    :param data
    encode Y1
    """
    y_temp = [1 if row['aspect_category'] == aspect_category else 0 for _, row in
              data.iterrows()]
    y = y_temp.copy()
    for index, value in enumerate(y_temp):
        if value == 1:
            text = data.text[index]
            for index, review in enumerate(data.text):
                if review == text:
                    y[index] = 1
    return y


# define default ap_list
ap_list = ['FOOD#QUALITY', 'FOOD#PRICES', 'FOOD#STYLE_OPTIONS', 'RESTAURANT#GENERAL', 'RESTAURANT#PRICES',
           'RESTAURANT#MISCELLANEOUS', 'DRINKS#PRICES', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
           'AMBIENCE#GENERAL', 'SERVICE#GENERAL', 'LOCATION#GENERAL']
# define file


train_file = '../data/official_data/ABSA16_Restaurants_Train_SB1_v2.xml'
train_csv = '../data/official_data/data_train.csv'
sample_csv = '../data/official_data/data_sample.csv'
test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'
test_csv = '../data/official_data/data_test.csv'
vocab_file = '../data/vocab_ap.txt'
ap_most_word = '../data/official_data/aspect_category_most_common_word'
embedding_file = '../data/glove.6B.100d.txt'
res_embedding_file = '../data/restaurant_emb.vec'
negative_words = '../data/negative-words.txt'
positive_words = '../data/positive-words.txt'
model_file_name = 'model_invidual_ap_classifier'
model_folder = '../data/model'

if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(model_folder + '/' + model_file_name):
    os.makedirs(model_folder + '/' + model_file_name)

data_train = pd.read_csv(train_csv, sep='\t')
data_test = pd.read_csv(test_csv, sep='\t')
data_sample = pd.read_csv(sample_csv, sep='\t')

vocab = helpers.load_doc(vocab_file)
vocab = set(vocab.split())

vocab_most_common = load_most_common_word(ap_most_word, ap_list)
# vocab_positive = helpers.load_doc(positive_words)
# vocab_positive = set(vocab_positive.split())
#
# vocab_negative = helpers.load_doc(negative_words)
# vocab_negative = set(vocab_negative.split())


# get aspect_category_list

# aspect_category_list = data_train.aspect_category.unique()
aspect_category_list = ['FOOD#QUALITY', 'RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL']
# aspect_category_list = ['FOOD#QUALITY']

X_dict_list = prepare_X_dict(data_train, vocab, vocab_most_common)
# print(X_dict_list[3]["transform_function"](data_sample.text))
train(X_dict_list, data_train, data_test)
model_list = load_model_list()
evaluate_model_list(model_list, X_dict_list, data_test)
