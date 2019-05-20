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


# X3, all Noun, ADJ from text

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


# X4, not, 0 1 vector

X4_clean_text = X1_clean_text

X4_process_texts = X1_process_texts


def X4_encode(text_array, max_length):
    temp = list()
    for text in text_array:
        init_array = [0] * max_length
        for index, value in enumerate(text.split()):
            if value == "not":
                init_array[index] = -1
        temp.append(init_array)
    return np.array(temp)


# X5, sentiment from dictionary
X5_clean_text = X4_clean_text
X5_process_texts = X4_process_texts


def X5_encode(text_array, vocab_negative, vocab_positive, max_length):
    temp = list()
    for text in text_array:
        init_array = [0] * max_length
        for index, value in enumerate(text.split()):
            if value in vocab_positive:
                init_array[index] = 1
            elif value in vocab_negative:
                init_array[index] = -1
        temp.append(init_array)
    return np.array(temp)


# X6 merge np array from other
def X6_encode(X_array):
    """:param X array
    merge all same X np
    """
    temp = list()
    for X in X_array:
        temp.append(np.expand_dims(X, 2))
    return np.dstack(temp)


# get all input, output X,Y
def prepare_X_dict(data_train, vocab, vocab_negative, vocab_positive):
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
    print(X1_max_length)
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

    # X3, Noun, Adj bag of word
    X3_train_texts = X3_process_texts(data_train.text, vocab)
    X3_tokenizer = create_tokenizer(X3_train_texts)
    X3_max_length = len(X3_tokenizer.word_index) + 1
    print(X3_max_length)
    def X3_transform_text_array(text_array):
        X_data = X3_process_texts(text_array, vocab)
        X_data = X3_encode(X_data, X3_tokenizer)
        return X_data

    X3_dict = {"max_length": X3_max_length, "transform_function": X3_transform_text_array}
    x_dict.append(X3_dict)

    # X4, not, 0 1 vector
    X4_train_texts = X4_process_texts(data_train.text, vocab)
    X4_max_length = max([len(s.split()) for s in X4_train_texts])

    def X4_transform_text_array(text_array):
        X_data = X4_process_texts(text_array, vocab)
        X_data = X4_encode(X_data, X4_max_length)
        return X_data

    X4_dict = {"max_length": X4_max_length, "transform_function": X4_transform_text_array}

    # X5, sentiment -1,1,0 vector
    X5_train_texts = X5_process_texts(data_train.text, vocab)
    X5_max_length = max([len(s.split()) for s in X5_train_texts])

    def X5_transform_text_array(text_array):
        X_data = X5_process_texts(text_array, vocab)
        X_data = X5_encode(X_data, vocab_negative, vocab_positive, X5_max_length)
        return X_data

    X5_dict = {"max_length": X5_max_length, "transform_function": X5_transform_text_array}

    # X6, 2 channel, not and sentiment from X4, X5
    X6_max_length = X4_max_length
    X6_list_dict = [X4_dict, X5_dict]
    X6_chanels_num = len(X6_list_dict)

    def X6_transform_text_array(text_array):
        temp = list()
        for X_dict in X6_list_dict:
            temp.append(X_dict["transform_function"](text_array))
        return X6_encode(temp)

    X6_dict = {"chanels_num": X6_chanels_num, "max_length": X6_max_length,
               "transform_function": X6_transform_text_array}

    x_dict.append(X6_dict)
    return x_dict


def filter_data_with_ap(ap, data):
    temp_csv = data[data.aspect_category.str.contains(ap)].reset_index()
    return temp_csv


def prepare_Y_dict(data_train, ap_list):
    y_dict = {}
    for ap in ap_list:
        label_encoder = LabelEncoder()
        label_encoder.fit(data_train.polarity)

        def Y_encode(polarity_array):
            label_encoder.transform(polarity_array)
            return to_categorical(label_encoder.transform(polarity_array))

        def Y_decode(categorical_array):
            return label_encoder.inverse_transform(categorical_array)

        y_dict[ap] = {"encoder": Y_encode, "decoder": Y_decode}
    return y_dict


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
    X1_drop = Dropout(0.5)(X1_conv)
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
    X2_conv = Conv1D(filters=300, kernel_size=5, activation='relu')(X2_embedding)
    X2_drop = Dropout(0.5)(X2_conv)
    X2_pool = MaxPooling1D(pool_size=2)(X2_drop)
    X2_flat = Flatten()(X2_pool)
    X2_output = X2_flat
    # X3
    X3 = x_dict_list[2]
    X3_max_length = X3["max_length"]
    X3_input = Input(shape=(X3_max_length,))
    X3_output = X3_input

    # X6
    X6 = x_dict_list[3]
    X6_max_length = X6["max_length"]
    X6_chanels_num = X6["chanels_num"]
    X6_input = Input(batch_shape=(None, X6["max_length"], X6_chanels_num))
    X6_conv = Conv1D(filters=100, kernel_size=3, activation='relu')(X6_input)
    X6_drop = Dropout(0.5)(X6_conv)
    X6_flat = Flatten()(X6_drop)
    X6_output = X6_flat

    # model
    merged = concatenate([X1_output, X2_output, X3_output, X6_output])
    dense1 = Dense(512, activation='relu')(merged)
    dense2 = Dense(10, activation='relu')(dense1)
    outputs = Dense(3, activation='sigmoid')(dense2)
    model = Model(inputs=[X1_input, X2_input, X3_input, X6_input], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
    return model

    # X3


def train(x_dict_list, y_dict, data_train, data_test):
    for ap in aspect_category_list:
        print(ap)
        data_train_ap = filter_data_with_ap(ap, data_train)
        data_test_ap = filter_data_with_ap(ap, data_test)
        model = define_model(x_dict_list)
        model.summary()
        plot_model(model, show_shapes=True, to_file=model_folder + '/' + model_file_name + '/' + ap + '.png')
        Y_train = y_dict[ap]["encoder"](data_train_ap.polarity)
        X_train = [X["transform_function"](data_train_ap.text) for X in x_dict_list]
        model.fit(X_train, Y_train, epochs=100, verbose=2)
        # evaluate_model(model, ap, x_dict_list, data_test)
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


def evaluate_model(model, aspect_category, x_dict_list, y_dict, data_test):
    data_test_ap = filter_data_with_ap(aspect_category, data_test)
    Y_test = y_dict[aspect_category]["encoder"](data_test_ap.polarity)
    X_test_array = [X["transform_function"](data_test_ap.text) for X in x_dict_list]
    _, acc, f1_score, precision, recall = model.evaluate(X_test_array, Y_test, verbose=2)
    print('%s Accuracy: %f' % (aspect_category, acc * 100))
    print('%s F1_score: %f' % (aspect_category, f1_score * 100))
    print('%s Precision: %f' % (aspect_category, precision * 100))
    print('%s Recall: %f' % (aspect_category, recall * 100))


def evaluate_model_list(model_list, x_dict_list, y_dict, data_test):
    for model in model_list:
        evaluate_model(model['model'], model['aspect_category'], x_dict_list, y_dict, data_test)


def predict(text_array, x_dict_list, decoder, model):
    text_predict = [X["transform_function"](text_array) for X in x_dict_list]
    y_hat = model.predict(text_predict)
    return decoder(y_hat.argmax(axis=-1)), y_hat[0]


def predict_with_ap(ap, text_array, x_dict_list, y_dict, model_list):
    for model in model_list:
        if model["aspect_category"] == ap:
            return predict(text_array, x_dict_list, y_dict[ap]["decoder"], model["model"])


def predict_input(x_dict_list, y_dict):
    model_list = load_model_list()
    ap = "FOOD#QUALITY"
    while True:
        inputText = input('nhap text: !!! \n')
        predicted = predict_with_ap(ap, [inputText], x_dict_list, y_dict, model_list)
        print(predicted)


def predict_outside(text_array, ap):
    return predict_with_ap(ap, text_array, X_dict_list, Y_dict, model_list)


# define file
train_file = '../data/official_data/ABSA16_Restaurants_Train_SB1_v2.xml'
train_csv = '../data/official_data/data_train.csv'
sample_csv = '../data/official_data/data_sample.csv'
test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'
test_csv = '../data/official_data/data_test.csv'
vocab_file = '../data/vocab.txt'
embedding_file = '../data/glove.6B.100d.txt'
res_embedding_file = '../data/restaurant_emb.vec'
negative_words = '../data/negative-words.txt'
positive_words = '../data/positive-words.txt'
model_file_name = 'model_invidual_sentiment_ap_classifier'
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

vocab_positive = helpers.load_doc(positive_words)
vocab_positive = set(vocab_positive.split())

vocab_negative = helpers.load_doc(negative_words)
vocab_negative = set(vocab_negative.split())
# aspect_category_list = ['DRINKS#PRICES']
aspect_category_list = data_train.aspect_category.unique()

X_dict_list = prepare_X_dict(data_train, vocab, vocab_negative, vocab_positive)
Y_dict = prepare_Y_dict(data_train, aspect_category_list)

train(X_dict_list, Y_dict, data_train, data_test)
model_list = load_model_list()
evaluate_model_list(model_list, X_dict_list, Y_dict, data_test)
predict_input(X_dict_list, Y_dict)
