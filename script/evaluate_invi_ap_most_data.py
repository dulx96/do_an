import individual_ap_classifier_most_data as ap
import xml.etree.ElementTree as ET

test_file = '../data/official_data/EN_REST_SB1_TEST_gold.xml'


def predict_obj_ap_from_file(file, threshold=25):
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
        predicted = ap.predict_outside([obj['text']])
        ap_list = [e['ap'] for e in predicted if e['H'] >= threshold]
        obj['ap'] = ap_list
        list_obj.append(obj)
    return list_obj


def gen_obj_ap_from_xml_file(file):
    """
    get object list(text, ap_list)
    :param file: xml ex file
    :return:
    """
    list_obj = []
    sentences = ET.parse(file).getroot().findall('./Review/sentences/sentence')
    for sentence in sentences:
        obj = {}
        obj['text'] = sentence.find('text').text
        Opinions = sentence.findall('./Opinions/Opinion')
        ap_list = []
        for Opinion in Opinions:
            ap_list.append(Opinion.get('category'))
        ap_list = list(dict.fromkeys(ap_list))
        obj['ap'] = ap_list
        list_obj.append(obj)
    return list_obj


def evaluate_mirco_ap(test, predicted):
    """
    :param test: list obj test
    :param predicted: list obj predicted
    :return:
    """
    tp = 0
    fp = 0
    # actually positive
    relevant = 0

    for index, sentence in enumerate(test):
        relevant += len(sentence['ap'])
    for index, predicted_sentence in enumerate(predicted):
        print(test[index]['text'])
        print(predicted_sentence['ap'])
        print(test[index]['ap'])
        for ap in predicted_sentence['ap']:
            if ap in test[index]['ap']:
                tp += 1
            else:
                fp += 1
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / relevant
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0
    return tp, fp, relevant, p, r, f1


print(evaluate_mirco_ap(gen_obj_ap_from_xml_file(test_file), predict_obj_ap_from_file(test_file)))
