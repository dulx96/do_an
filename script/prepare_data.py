import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
import re
import string
from nltk.corpus import stopwords
import nltk

train_file = '../data/official_data/ABSA16_Restaurants_Train_SB1_v2.xml'
train_csv = '../data/official_data/data_train.csv'
test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'
test_csv = '../data/official_data/data_test.csv'
vocab_file = '../data/vocab.txt'
ap_file_train = '../data/official_data/aspect_category/train'
ap_file_test = '../data/official_data/aspect_category/test'
ap_most_word = '../data/official_data/aspect_category_most_common_word'
ap_most_word_test = '../data/official_data/aspect_category_most_common_word_test'
vocab_ap_file = '../data/vocab_ap.txt'
vocab_sentiment_file = '../data/vocab_sentiment.txt'
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


def add_doc_to_vocab_task1(file, vocab):
    data = pd.read_csv(file, sep='\t')
    for text in data.text:
        tokens = helpers.clean_text_to_tokens_4(text)
        vocab.update(tokens)

def add_doc_to_vocab_task2(file, vocab):
    data = pd.read_csv(file, sep='\t')
    for text in data.text:
        tokens = helpers.clean_text_to_tokens_3(text)
        vocab.update(tokens)

def save_vocab(tokens, file):
    data = '\n'.join(tokens)
    file = open(file, 'w')
    file.write(data)
    file.close()


def gen_vocab_ap(file_vocab, file_data):
    vocab = Counter()
    min_occurence = 1
    add_doc_to_vocab_task1(file_data, vocab)
    tokens = [k for k, c in vocab.items() if c >= min_occurence]
    save_vocab(tokens, file_vocab)

def gen_vocab_sentiment(file_vocab, file_data):
    vocab = Counter()
    min_occurence = 1
    add_doc_to_vocab_task2(file_data, vocab)
    tokens = [k for k, c in vocab.items() if c >= min_occurence]
    save_vocab(tokens, file_vocab)

def count_data(data, ap_list, polarity_list):
    for ap in ap_list:
        sum = 0
        for polarity in polarity_list:
            print(ap + polarity)
            temp = data[data.aspect_category.str.contains(ap) & data.polarity.str.contains(polarity)].reset_index()
            print(len(temp))
            sum += len(temp)
        print(sum)


def filter_data_with_ap(ap, data):
    # ap = ap.split('#')[0]
    temp_csv = data[data.aspect_category.str.contains(ap)].reset_index()
    return temp_csv


def gen_ap_file(data, ap_list, file):
    for ap in ap_list:
        output = file + '/' + ap + '.csv'
        data_ap = filter_data_with_ap(ap, data)
        data_ap.to_csv(output, sep='\t', index=False, header=True)


def gen_most_common_word_in_ap(data, ap_list, file):
    is_noun = lambda pos: pos[:2] == 'NN'
    is_adj = lambda pos: pos[:2] == 'JJ'
    for ap in ap_list:
        print(ap)
        output = file + '/' + ap + '.txt'
        data_ap = filter_data_with_ap(ap, data)
        vocab_noun = Counter()
        vocab_adj = Counter()
        for text in data_ap.text:
            tokens = helpers.clean_text_to_tokens_3(text)
            vocab_noun.update([word for (word, pos) in nltk.pos_tag(tokens) if is_noun(pos)])
            vocab_adj.update([word for (word, pos) in nltk.pos_tag(tokens) if is_adj(pos)])
        most_common_noun = vocab_noun.most_common(10)
        most_common_adj = vocab_adj.most_common(5)
        most_common_tokens = [m[0] for m in most_common_noun] + [m[0] for m in most_common_adj]
        save_vocab(most_common_tokens, output)


ap_list = ['FOOD#QUALITY', 'FOOD#PRICES', 'FOOD#STYLE_OPTIONS', 'RESTAURANT#GENERAL', 'RESTAURANT#PRICES',
           'RESTAURANT#MISCELLANEOUS', 'DRINKS#PRICES', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
           'AMBIENCE#GENERAL', 'SERVICE#GENERAL', 'LOCATION#GENERAL']
data_train = pd.read_csv(train_csv, sep='\t')
data_test = pd.read_csv(test_csv, sep='\t')
# gen_ap_file(data_train, ap_list, ap_file_train)
# gen_ap_file(data_test, ap_list, ap_file_test)
# count_data(data_test13, ['FOOD#QUALITY', 'FOOD#PRICES', 'FOOD#STYLE_OPTIONS', 'RESTAURANT#GENERAL', 'RESTAURANT#PRICES',
#                         'RESTAURANT#MISCELLANEOUS', 'DRINKS#PRICES', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
#                         'AMBIENCE#GENERAL', 'SERVICE#GENERAL','LOCATION#GENERAL'], ['positive', 'negative', 'neutral'])

gen_vocab_ap(vocab_ap_file, train_csv)
gen_vocab_sentiment(vocab_sentiment_file, train_csv)
# to_csv(test_file, test_csv)
# gen_most_common_word_in_ap(data_test, ap_list, ap_most_word_test)
# gen_most_common_word_in_ap(data_train, ap_list, ap_most_word)
