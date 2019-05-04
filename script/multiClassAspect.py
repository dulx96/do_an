import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import spacy
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Flatten, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

import helpers

pd.option_context('display.max_rows', 5, 'display.max_columns', 5)

import pandas as pd


# aspect_terms = []
# for review in nlp.pipe(dataset.text):
#     chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
#     aspect_terms.append(' '.join(chunks))
# dataset['aspect_terms'] = aspect_terms

def clean_text(text, vocab):
    tokens = helpers.clean_text_to_tokens(text)
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    texts = ' '.join(tokens)
    return texts


def process_texts(text_array, vocab):
    texts_clean_list = list()
    for text in text_array:
        texts_clean_list.append(clean_text(text, vocab))
    return texts_clean_list


def encode_X(text_array, tokenizer, max_length):
    encoded = tokenizer.texts_to_sequences(text_array)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


def encode_Y(aspect_category, data):
    y_temp = [1 if n == aspect_category else 0 for n in data.aspect_category]
    y = y_temp.copy()
    for index, value in enumerate(y_temp):
        if value == 1:
            text = data.text[index]
            for index, review in enumerate(data.text):
                if review == text:
                    y[index] = 1
    return y


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


def define_model(vocab_size, max_length, embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(X_train, aspect_category_list, vocab_size, max_length, embedding_matrix):
    for aspect_category in aspect_category_list:
        y_train = encode_Y(aspect_category, data_train)
        model = define_model(vocab_size, max_length, embedding_matrix)
        model.fit(X_train, y_train, epochs=5, verbose=1)
        model.save('../model/' + aspect_category + 'model.h5')


def load_model_aspect():
    model = []
    for aspect_category in aspect_category_list:
        model.append(
            {'aspect_category': aspect_category, 'model': load_model('../model/' + aspect_category + 'model.h5')})
    return model


def predict(text, model_list, vocab, tokenizer, max_length):
    text_clean_array = process_texts([text], vocab)
    print(text_clean_array)
    X_predict = encode_X(text_clean_array, tokenizer, max_length)
    print(X_predict)
    result = []
    for model in model_list:
        y_hat = model['model'].predict(X_predict, verbose=0)
        percent_tag = y_hat[0, 0]
        if round(percent_tag * 100) <= 30:
            print(model['aspect_category'], percent_tag * 100)
            continue
        print(model['aspect_category'], percent_tag * 100)
        result.append(model['aspect_category'])
    return result


def evaluate(X_test, model_list):
    for model in model_list:
        y_test = encode_Y(model['aspect_category'], data_test)
        print(y_test[0])
        _, acc = model['model'].evaluate(X_test, y_test, verbose=1)
        print('Train %s accuracy: %f' % (model['aspect_category'], acc * 100))


def create_tokenizer(text_array):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_array)
    return tokenizer


# define file
train_file = '../data/official_data/ABSA16_Restaurants_Train_SB1_v2.xml'
train_csv = '../data/official_data/data_train.csv'
test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'
test_csv = '../data/official_data/data_test.csv'
vocab_file = '../data/vocab.txt'
embedding_file = '../data/glove.6B.100d.txt'

data_train = pd.read_csv(train_csv, sep='\t')
data_test = pd.read_csv(test_csv, sep='\t')
# nlp = spacy.load('en')
# dataset['text'] = dataset.text.str.lower()
# data_test['text'] = data_test.text.str.lower()

vocab = helpers.load_doc(vocab_file)
vocab = set(vocab.split())
# init
train_texts = process_texts(data_train.text, vocab)
test_texts = process_texts(data_test.text, vocab)
tokenizer = create_tokenizer(train_texts)


vocab_size = len(tokenizer.word_index) + 1
print(' Vocabulary size: %d ' % vocab_size)
max_length = max([len(s.split()) for s in train_texts])
print(' Maximum length: %d ' % max_length)

# generate Xtrain, Xtest
X_train = encode_X(train_texts, tokenizer, max_length)
X_test = encode_X(test_texts, tokenizer, max_length)

embedding_matrix = get_pretrained_embedding(embedding_file, tokenizer, vocab_size)
# aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(dataset.aspect_terms))
# X_train = encode_X(dataset.text, tokenizer)

# get aspect_category_list
aspect_category_list = data_train.aspect_category.unique()
# TRAIN
# train(X_train, aspect_category_list, vocab_size, max_length, embedding_matrix)

# Evaluate
model_list = load_model_aspect()
evaluate(X_test, model_list)
#
# predict
while True:
    inputText = input('nhap text: !!! \n')
    predicted = predict(inputText, model_list, vocab, tokenizer, max_length)
    print(predicted)
