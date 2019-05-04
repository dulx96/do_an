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


def clean_text(text, vocab):
    tokens = helpers.clean_text_to_tokens(text)
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    texts = ' '.join(tokens)
    return texts


def clean_text_pos(text):
    tokens = nltk.word_tokenize(text)
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def process_texts(text_array, vocab):
    texts_clean_list = list()
    for text in text_array:
        texts_clean_list.append(clean_text(text, vocab))
    return texts_clean_list


def process_texts_pos(text_array):
    tags_list = list()
    for text in text_array:
        # text_clean_list.append(clean_text_pos(text))
        tagged = nltk.pos_tag(clean_text_pos(text))
        tags = [tagged[i][1] for i, _ in enumerate(tagged)]
        tags_list.append(tags)
    return tags_list


def create_tokenizer(text_array):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_array)
    return tokenizer


def encode_X1(text_array, tokenizer, max_length):
    encoded = tokenizer.texts_to_sequences(text_array)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


def encode_Y1(data_array, encoder):
    return to_categorical(encoder.transform(data_array))


def decode_Y1(data_array, encoder):
    return encoder.inverse_transform(data_array)


def encode_X2(data_array, encoder):
    return encoder.transform(data_array.reshape(len(data_array), 1))


def encode_X3(data_array, tokenizer, max_length):
    encoded = tokenizer.texts_to_sequences(data_array)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return np.expand_dims(padded, 2)


# def encode_Y():
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

# init
train_texts = process_texts(data_train.text, vocab)
test_texts = process_texts(data_test.text, vocab)

tokenizer = create_tokenizer(train_texts)
vocab_size = len(tokenizer.word_index) + 1
print(' Vocabulary size: %d ' % vocab_size)
max_length = max([len(s.split()) for s in train_texts])
print(' Maximum length: %d ' % max_length)

aspect_category_list = data_train.aspect_category.unique()
# X1, review w2v
X1_train = encode_X1(train_texts, tokenizer, max_length)
X1_test = encode_X1(test_texts, tokenizer, max_length)
X1_embedding_matrix = get_pretrained_embedding(embedding_file, tokenizer, vocab_size)
# Y1, polarity
y1_label_encoder = LabelEncoder()
y1_label_encoder.fit(data_train.polarity)
Y1_train = encode_Y1(data_train.polarity, y1_label_encoder)
Y1_test = encode_Y1(data_test.polarity, y1_label_encoder)

# X2, aspect
x2_encoder = OneHotEncoder(sparse=False)
x2_encoder.fit(aspect_category_list.reshape(len(aspect_category_list), 1))
X2_train = encode_X2(data_train.aspect_category.values, x2_encoder)
X2_test = encode_X2(data_test.aspect_category.values, x2_encoder)

# X3, pos
X3_train_data = process_texts_pos(data_train.text)
X3_test_data = process_texts_pos(data_test.text)
# max_length_x3 = max([len(i) for i in X3_train_data])
# all_tag = np.array(
#     ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS",
#      "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT",
#      "WP",
#      "WP$", "WRB"])
# all_tag = all_tag.reshape(len(all_tag), 1)
# x3_encoder.fit(all_tag)
x3_tokenizer = create_tokenizer(X3_train_data)
x3_max_length = max([len(s) for s in X3_train_data])

# X3_train = x3_tokenizer.texts_to_matrix(X3_train_data, mode='binary')
# X3_test = x3_tokenizer.texts_to_matrix(X3_test_data, mode='binary')
X3_train = encode_X3(X3_train_data, x3_tokenizer, x3_max_length)
X3_test = encode_X3(X3_test_data, x3_tokenizer, x3_max_length)

print(X3_train)
# X4, tfidf
tf_idf_tokenizer = create_tokenizer(train_texts)
X4_train = tf_idf_tokenizer.texts_to_matrix(train_texts, mode='tfidf')
X4_test = tf_idf_tokenizer.texts_to_matrix(test_texts, mode='tfidf')
x4_length = X4_train.shape[1]

# review input
inputs1 = Input(shape=(max_length,))
embedding1 = Embedding(vocab_size, 100, weights=[X1_embedding_matrix])(inputs1)
conv1 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)

# aspect input
inputs2 = Input(shape=(12,))

# pos input
inputs3 = Input(batch_shape=(None, x3_max_length, 1))
# inputs3_1 = Dense(50, activation='relu')(inputs3)
conv3 = Conv1D(filters=32, kernel_size=2, activation='relu')(inputs3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)
# tfidf input
inputs4 = Input(shape=(x4_length,))

# 1 more cnn
inputs5 = Input(shape=(max_length,))
embedding5 = Embedding(vocab_size, 100, weights=[X1_embedding_matrix])(inputs5)
conv5 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding5)
drop5 = Dropout(0.5)(conv5)
pool5 = MaxPooling1D(pool_size=2)(drop5)
flat5 = Flatten()(pool5)

# merge
merged = concatenate([flat1, inputs2, flat3, flat5])
dense1 = Dense(512, activation='relu')(merged)
dense2 = Dense(10, activation='relu')(dense1)
outputs = Dense(3, activation='softmax')(dense2)

model = Model(inputs=[inputs1, inputs2, inputs3, inputs5], outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
plot_model(model, show_shapes=True, to_file='model.png')
model.fit([X1_train, X2_train, X3_train, X1_train], Y1_train, epochs=200, verbose=2)
_, acc = model.evaluate([X1_test, X2_test, X3_test, X1_test], Y1_test, verbose=2)
print('Train accuracy: %f' % (acc * 100))


def load_model_sentiment():
    return load_model('sentimen_model.h5')


def load_model_aspect(aspect_category_list):
    model = []
    for aspect_category in aspect_category_list:
        model.append(
            {'aspect_category': aspect_category, 'model': load_model('../model/' + aspect_category + 'model.h5')})
    return model


def encode_aspect_category(text, tokenizer):
    chunks = [(chunk.root.text) for chunk in nlp(text).noun_chunks if chunk.root.pos_ == 'NOUN']
    new_review_aspect_terms = ' '.join(chunks)
    new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])
    return new_review_aspect_tokenized


def predict_aspect_category(text, model_list, tokenizer):
    # model_list = load_model()
    encoded_X = encode_aspect_category(text, tokenizer)
    result = []
    for model in model_list:
        y_hat = model['model'].predict(encoded_X, verbose=0)
        percent_tag = y_hat[0, 0]
        if round(percent_tag * 100) <= 30:
            print(model['aspect_category'], percent_tag * 100)
            continue
        print(model['aspect_category'], percent_tag * 100)
        result.append(model['aspect_category'])
    return result


def encode_sentiment_text(text, tokenizer):
    sentiment = []
    for review in nlp.pipe([text]):
        if review.is_parsed:
            sentiment.append(' '.join([token.lemma_ for token in review if (
                    not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
        else:
            sentiment.append('')
    return tokenizer.texts_to_matrix([sentiment])


def encode_aspect(ap, list):
    enc = OneHotEncoder(sparse=False)
    onehot_encoded = enc.fit(list.reshape(len(list), 1))
    return onehot_encoded.transform([[ap]])


def predict_sentiment(text, ap, model):
    text = encode_sentiment_text(text, tokenizer)
    ap = encode_aspect(ap, aspect_category_list)
    y_hat = model.predict([text, ap])
    x, y, z = y_hat[0]
    x = x * 100
    y = y * 100
    z = z * 100
    if x > y and x > z:
        return [1.0, 0, 0]
    if y > x and y > z:
        return [0, 1.0, 0]
    if z > x and z > y:
        return [0, 0, 1.0]


# model_ap_list = load_model_aspect(aspect_category_list)
# model_sentiment = load_model_sentiment()


def inverse_sentiment(array):
    if array[0] == 1.0:
        return 'NEGATIVE'
    if array[1] == 1.0:
        return 'POSITIVE'
    if array[2] == 1.0:
        return 'NEUTRAL'

#
# while True:
#     inputText = input('nhap text: !!! \n')
#     ap_predicted = predict_aspect_category(inputText, model_ap_list, tokenizer)
#     for ap in ap_predicted:
#         sentiment_predicted = predict_sentiment(inputText, ap, model_sentiment)
#         print(ap, inverse_sentiment(sentiment_predicted))
