import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
import re
import string
from nltk.corpus import stopwords

train_file = '../data/official_data/ABSA16_Restaurants_Train_SB1_v2.xml'
train_csv = '../data/official_data/data_train.csv'
test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'
test_csv = '../data/official_data/data_test.csv'
vocab_file = '../data/vocab.txt'

import helpers
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


def clean_text_to_tokens(text):
    text_lower = text.lower()
    tokens = text_lower.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # # filter out stop words
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(file, vocab):
    data = pd.read_csv(file, sep='\t')
    for text in data.text:
        tokens = helpers.clean_text_to_tokens_2(text)
        vocab.update(tokens)


def save_vocab(tokens, file):
    data = '\n'.join(tokens)
    file = open(file, 'w')
    file.write(data)
    file.close()


def gen_vocab(file_vocab, file_data):
    vocab = Counter()
    min_occurence = 1
    add_doc_to_vocab(file_data, vocab)
    tokens = [k for k, c in vocab.items() if c >= min_occurence]
    print(tokens)
    save_vocab(tokens, file_vocab)


gen_vocab(vocab_file, train_csv)
# to_csv(test_file, test_csv)
