import individual_sentiment_with_ap_multi_classifier as sentiment

import xml.etree.ElementTree as ET

from tensorflow.python.util.tf_export import api_export

test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'


def gen_obj_ap_from_xml_file(file):
    """
    get object list(text, [{ap, sentiment}])
    :param file: xml ex file
    :return:
    """
    list_obj = []
    sentences = ET.parse(file).getroot().findall('./Review/sentences/sentence')
    for sentence in sentences:
        obj = {}
        obj['text'] = sentence.find('text').text
        Opinions = sentence.findall('./Opinions/Opinion')
        ap_sentiment_obj = {}
        for Opinion in Opinions:
            ap_sentiment_obj[Opinion.get('category')] = Opinion.get('polarity')

        obj['sentiment'] = ap_sentiment_obj
        list_obj.append(obj)
    return list_obj


def predict_obj_ap_from_file(file):
    """
    get object list(text, ap_list)
    :param file:
    :return:
    """
    list_obj = []
    sentences = ET.parse(file).getroot().findall('./Review/sentences/sentence')
    for sentence in sentences:
        obj = {}
        obj['text'] = sentence.find('text').text
        Opinions = sentence.findall('./Opinions/Opinion')
        ap_sentiment_obj = {}
        for Opinion in Opinions:
            ap = Opinion.get('category')
            if not ap == 'DRINKS#PRICES':
                ap_sentiment_obj[ap] = 'positive'
            else:
                predicted = sentiment.predict_outside([obj['text']], ap)
                ap_sentiment_obj[ap] = predicted[0]
        obj['sentiment'] = ap_sentiment_obj
        list_obj.append(obj)
    return list_obj


def evaluate_mirco_ap(test, predicted):
    """
    :param test: list obj test
    :param predicted: list obj predicted
    :return:
    """
    true = 0
    total = 0
    # actually positive
    # relevant = 0

    for index, sentence in enumerate(test):
        total += len(sentence['sentiment'])
    for index, predicted_sentence in enumerate(predicted):
        print(predicted_sentence['text'])
        print(test[index]['sentiment'])
        print(predicted_sentence['sentiment'])
        for k, v in dict.items(predicted_sentence['sentiment']):
            if test[index]['sentiment'][k] == v:
                true += 1
    return total, true


print(evaluate_mirco_ap(gen_obj_ap_from_xml_file(test_file), predict_obj_ap_from_file(test_file)))
