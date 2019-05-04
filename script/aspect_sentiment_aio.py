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


def X1_clean_text(text, vocab):
    tokens = helpers.clean_text_to_tokens(text)
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


def X2_clean_text(text, vocab):
    tokens = helpers.clean_text_to_tokens(text)
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = helpers.convert_contracted_form_negative(tokens)
    texts = ' '.join(tokens)
    return texts


def X2_process_texts(text_array, vocab):
    texts_clean_list = list()
    for text in text_array:
        texts_clean_list.append(X2_clean_text(text, vocab))
    return texts_clean_list


def X2_encode(text_array, max_length):
    temp = list()
    for text in text_array:
        init_array = [0] * max_length
        for index, value in enumerate(text.split()):
            if value == "not":
                init_array[index] = 1
        temp.append(init_array)
    return np.array(temp)


def X3_encode(text_array, tokenizer, max_length):
    encoded = tokenizer.texts_to_sequences(text_array)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


def X6_encode(X2, X3):
    X2 = np.expand_dims(X2, 2)
    X3 = np.expand_dims(X3, 2)
    print(X2.shape)
    print(X3.shape)
    temp = np.dstack((X2, X3))
    return temp


def Y1_encode(aspect_category, polarity, data):
    y_temp = [1 if row['aspect_category'] == aspect_category and row.polarity == polarity else 0 for _, row in
              data.iterrows()]
    y = y_temp.copy()
    for index, value in enumerate(y_temp):
        if value == 1:
            text = data.text[index]
            for index, review in enumerate(data.text):
                if review == text:
                    y[index] = 1
    return y


def create_tokenizer(text_array):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_array)
    return tokenizer


def get_pretrained_embedding(file, tokenizer, vocab_size):
    embedding_index = dict()
    f = open(file, mode='rt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def define_model(X1_vocab_size, X1_max_length, X1_embedding_matrix, X6_max_length):
    # review input,w2v
    X1_input = Input(shape=(X1_max_length,))
    embedding1 = Embedding(X1_vocab_size, 100, weights=[X1_embedding_matrix])(X1_input)
    conv1 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    X1_output = flat1

    # review input,w2v
    X1_2_input = Input(shape=(X1_max_length,))
    embedding1_2 = Embedding(X1_vocab_size, 100, weights=[X1_embedding_matrix])(X1_2_input)
    conv1_2 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding1_2)
    drop1_2 = Dropout(0.5)(conv1_2)
    pool1_2 = MaxPooling1D(pool_size=2)(drop1_2)
    flat1_2 = Flatten()(pool1_2)
    X1_2_ouput = flat1_2

    # # not input, 1 dimension vector
    # X2_input = Input(shape=(X2_max_length,))
    # # X2_dense = Dense(512, activation='relu')(X2_input)
    # X2_ouput = X2_input

    # not and sequence 2 channel vector
    X6_input = Input(batch_shape=(None, X6_max_length, 2))
    X6_conv = Conv1D(filters=32, kernel_size=2, activation='relu')(X6_input)
    X6_drop = Dropout(0.5)(X6_conv)
    X6_pool = MaxPooling1D(pool_size=2)(X6_drop)
    X6_flat = Flatten()(X6_pool)
    X6_output = X6_flat

    # merge
    mereged = concatenate([X1_output, X1_2_ouput, X6_output])
    dense1 = Dense(512, activation='relu')(mereged)
    dense2 = Dense(10, activation='relu')(dense1)
    outputs = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=[X1_input, X1_2_input, X6_input], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train():
    for ap in aspect_category_list:
        for polarity in ['positive', 'negative', 'neutral']:
            model = define_model(X1_vocab_size, X1_max_length, X1_embedding_matrix, X6_max_length)
            model.summary()
            plot_model(model, show_shapes=True, to_file='../model/' + ap + polarity + '.png')
            Y1_train = Y1_encode(ap, polarity, data_train)
            Y1_test = Y1_encode(ap, polarity, data_test)
            model.fit([X1_train, X1_train, X6_train], Y1_train, epochs=100, verbose=2)
            _, acc = model.evaluate([X1_test, X1_test, X6_test], Y1_test, verbose=2)
            print('%s Accuracy: %f' % (ap + polarity, acc * 100))
            model.save('../model/' + ap + '#' + polarity + 'model.h5')


def load_model_list():
    model = []
    for ap in aspect_category_list:
        for polarity in ['positive', 'negative', 'neutral']:
            model.append(
                {'aspect_category': ap, 'polarity': polarity,
                 'model': load_model('../model/' + ap + '#' + polarity + 'model.h5')}
            )
    return model


def predict(text, model_list):
    X1_predict = X1_process_texts([text], vocab)
    print(X1_predict)
    X1_predict = X1_encode(X1_predict, X1_tokenizer, X1_max_length)
    print(text)
    X2_predict = X2_process_texts([text], vocab)
    print(X2_predict)
    X2_predict = X2_encode(X2_predict, X2_max_length)
    X3_predict = X3_encode([text], X3_tokenizer, X3_max_length)
    X6_predict = X6_encode(X2_predict, X3_predict)
    result = []
    for model in model_list:
        y_hat = model['model'].predict([X1_predict, X1_predict, X6_predict], verbose=2)
        percent_tag = y_hat[0, 0]
        if round(percent_tag * 100) < 50:
            print(model['aspect_category'], '#', model['polarity'], percent_tag * 100)
            continue
        print(model['aspect_category'], '#', model['polarity'], percent_tag * 100)
        result.append(model['aspect_category'] + model['polarity'])
    return result


def evaluate(model_list):
    for model in model_list:
        Y1_test = Y1_encode(model['aspect_category'], model['polarity'], data_test)
        _, acc = model['model'].evaluate([X1_test, X1_test, X6_test], Y1_test, verbose=2)
        print('%s Accuracy: %f' % (model['aspect_category'] + '#' + model['polarity'], acc * 100))


def test_input():
    model_list = load_model_list()
    evaluate(model_list)
    while True:
        input_text = input('nhap text: !!! \n')
        predicted = predict(input_text, model_list)
        print(predicted)


# define file


train_file = '../data/official_data/ABSA16_Restaurants_Train_SB1_v2.xml'
train_csv = '../data/official_data/data_train.csv'
test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'
test_csv = '../data/official_data/data_test.csv'
vocab_file = '../data/vocab.txt'
embedding_file = '../data/glove.6B.100d.txt'

data_train = pd.read_csv(train_csv, sep='\t')
data_test = pd.read_csv(test_csv, sep='\t')

vocab = helpers.load_doc(vocab_file)
vocab = set(vocab.split())

# X1, review w2v
X1_train_texts = X1_process_texts(data_train.text, vocab)
X1_test_texts = X1_process_texts(data_test.text, vocab)
X1_tokenizer = create_tokenizer(X1_train_texts)
X1_vocab_size = len(X1_tokenizer.word_index) + 1
X1_max_length = max([len(s.split()) for s in X1_train_texts])
X1_embedding_matrix = get_pretrained_embedding(embedding_file, X1_tokenizer, X1_vocab_size)
X1_train = X1_encode(X1_train_texts, X1_tokenizer, X1_max_length)
X1_test = X1_encode(X1_test_texts, X1_tokenizer, X1_max_length)

# X2, not, boolean vector
X2_train_texts = X2_process_texts(data_train.text, vocab)
X2_test_texts = X2_process_texts(data_test.text, vocab)
X2_max_length = max([len(s.split()) for s in X2_train_texts])
X2_train = X2_encode(X2_train_texts, X2_max_length)
X2_test = X2_encode(X2_test_texts, X2_max_length)
print(X2_train)
# X3, tfidf with not, tokenizer to matrix, based on clean text from x2
X3_train_texts = X2_train_texts
X3_test_texts = X2_test_texts
X3_tokenizer = create_tokenizer(X3_train_texts)
X3_vocab_size = len(X3_tokenizer.word_index) + 1
X3_max_length = X2_max_length
X3_train = X3_encode(X3_train_texts, X3_tokenizer, X3_max_length)
X3_test = X3_encode(X3_test_texts, X3_tokenizer, X3_max_length)
print(X3_train)

# X6 merege to 2 dimension vector from x2,x3
X6_max_length = X2_max_length
X6_train = X6_encode(X2_train, X3_train)
X6_test = X6_encode(X2_test, X3_test)

# get aspect_category_list
aspect_category_list = data_train.aspect_category.unique()
# aspect_category_list = ['FOOD#QUALITY']

# train()
# model_list = load_model_list()
# evaluate(model_list)
test_input()
# print(X1_embedding_matrix)
# print(X1_train)
# print(Y1_train)
