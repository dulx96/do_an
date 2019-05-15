import re
import string
from nltk.corpus import stopwords
import nltk

# Stopwords, imported from NLTK (v 2.0.4)
stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'be', 'been',
     'being', 'have', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'only',
     'own', 'same', 'so', 'than', 's', 't', 'now'])


def load_doc(file):
    file = open(file, 'r')
    text = file.read()
    file.close()
    return text


# clean text heplers

def clean_text_to_tokens(text):
    text_lower = text.lower()
    tokens = text_lower.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out stop words
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def clean_text_4(text):
    """work token by nltk, punctuation, short word <= 1"""
    tokens = nltk.word_tokenize(text)
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def clean_text_to_tokens_1(text):
    """:param text:string
        lower, split by white space, remove non Aplpha, remove word length <=1
     """
    text_lower = text.lower()
    tokens = text_lower.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # # stopword remove
    # tokens = [w for w in tokens if not w in stopwords]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def clean_text_to_tokens_2(text):
    """
    :param text:string
    loweer, split by white space, remove not Alpha, remove word length <=1 , contracted not, stop_word
    :return:
    """
    text_lower = text.lower()
    tokens = text_lower.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # stopword remove
    tokens = [w for w in tokens if not w in stopwords]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = convert_contracted_form_negative(tokens)
    return tokens


def clean_text_to_tokens_3(text):
    """
    :param text:string
    split by white space , contracted not, join, split by nltk, remove not Alpha, remove word length <=1 , stop_word, stem
    :return:
    """
    tokens = text.split()
    tokens = convert_contracted_form_negative(tokens)
    tokens = ' '.join(tokens)
    tokens = nltk.word_tokenize(tokens)
    porter = nltk.stem.porter.PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stopwords]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def convert_contracted_form_negative(tokens):
    temp = list()
    contracted_form_1 = ["arent", "isnt", "wasnt", "werent", "couldnt", "cant", "mustnt", "shouldnt",
                         "wouldnt", "wont", "didnt", "doesnt", "dont", "hasnt", "havent", "hadnt"]
    contracted_form_2 = ["aren't", "isn't", "wasn't", "weren't", "couldn't", "can't", "mustn't", "shouldn't",
                         "wouldn't", "won't", "didn't", "doesn't", "don't", "hasn't", "haven't", "hadn't"]
    contracted_form_3 = ["aren’t", "isn’t", "wasn’t", "weren’t", "couldn’t", "can’t", "mustn’t", "shouldn’t",
                         "wouldn’t", "won’t", "didn’t", "doesn’t", "don’t", "hasn’t", "haven’t", "hadn’t"]
    for token in tokens:
        if token not in contracted_form_1 and token not in contracted_form_2 and token not in contracted_form_3:
            temp.append(token)
        else:
            if token == "wont" or token == "won't" or token == 'won’t':
                temp = temp + ['will', 'not']
            else:
                x = re.findall("(.*)nt", token)
                if len(x) >= 1:
                    temp.append(x[0])
                x = re.findall("(.*)n't", token)
                if len(x) >= 1:
                    temp.append(x[0])
                x = re.findall("(.*)n’t", token)
                if len(x) >= 1:
                    temp.append(x[0])
                temp.append('not')
    return temp
