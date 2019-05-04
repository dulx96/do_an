import pandas as pd
import xml.etree.ElementTree as ET
import spacy
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

pd.option_context('display.max_rows', 5, 'display.max_columns', 5)


# each opinion is a row
def parse_xml_to_pd(xml_file):
    out_df = pd.DataFrame(columns=['text', 'aspect_category', 'polarity'])
    if xml_file.endswith('.xml'):
        with open(xml_file) as f:
            dom = ET.parse(f)
            root = dom.getroot()
            for sentence in root.iter('sentence'):
                for opinion in sentence.iter('Opinion'):
                    text = sentence.find('text').text
                    aspect_category = opinion.get('category')
                    polarity = opinion.get('polarity')
                    out_df = out_df.append({'text': text, 'aspect_category': aspect_category, 'polarity': polarity},
                                           ignore_index=True)

    return out_df


def to_csv(file, output):
    data = parse_xml_to_pd(file)
    data.to_csv(output, sep='\t', index=False, header=True)


train_file = '../data/official_data/ABSA16_Restaurants_Train_SB1_v2.xml'
train_csv = '../data/official_data/data_train.csv'

test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'
test_csv = '../data/official_data/data_test.csv'
# generate csv from xml file
to_csv(test_file, test_csv)

dataset = pd.read_csv(train_csv, sep='\t')

# print(dataset.head(5))

# # get all noun as aspect term
# nlp = spacy.load('en')
# dataset['text'] = dataset.text.str.lower()
#
# aspect_terms = []
# for review in nlp.pipe(dataset.text):
#     chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
#     aspect_terms.append(' '.join(chunks))
# dataset['aspect_terms'] = aspect_terms
# # print(dataset.head(5))

# aspect_categories_model = Sequential()
# aspect_categories_model.add(Dense(512, input_shape=(6000,), activation='relu'))
# aspect_categories_model.add(Dense(12, activation='softmax'))
# aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #
# vocab_size = 6000
# tokenizer = Tokenizer(num_words=vocab_size)
# tokenizer.fit_on_texts(dataset.text)
# # aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(dataset.aspect_terms))
# label_encoder = LabelEncoder()
# integer_category = label_encoder.fit_transform(dataset.aspect_category)
# dummy_category = to_categorical(integer_category)
# aspect_categories_model.fit(aspect_tokenized, dummy_category, epochs=5, verbose=1)
# aspect_categories_model.save('aspect_category_model.h5')


# test review

# new_review = "Finally a meal that you will remember for a long time!   "
# aspect_categories_model = load_model('aspect_category_model.h5')
# chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']
# new_review_aspect_terms = ' '.join(chunks)
# new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])
# new_review_category = label_encoder.inverse_transform(aspect_categories_model.predict_classes(new_review_aspect_tokenized))
# print(new_review_category)

# sentiment part
# sentiment_terms = []
# for review in nlp.pipe(dataset.text):
#     if review.is_parsed:
#         sentiment_terms.append(' '.join([token.lemma_ for token in review if (
#                 not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
#     else:
#         sentiment_terms.append('')
# dataset['sentiment_terms'] = sentiment_terms
#
# sentiment_model = Sequential()
# sentiment_model.add(Dense(512, input_shape=(6000,), activation='relu'))
# sentiment_model.add(Dense(3, activation='softmax'))
# sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# sentiment_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(dataset.sentiment_terms))
#
# label_encoder_2 = LabelEncoder()
# integer_sentiment = label_encoder_2.fit_transform(dataset.polarity)
# dummy_sentiment = to_categorical(integer_sentiment)
#
# sentiment_model.fit(sentiment_tokenized, dummy_sentiment, epochs=5, verbose=1)
# sentiment_model.save('sentiment')
