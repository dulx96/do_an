import individual_sentiment_with_ap_multi_classifier as sentiment

import xml.etree.ElementTree as ET

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
        ap_sentiement_list = []
        temp_obj = {}
        for Opinion in Opinions:
            temp_obj[(Opinion.get('category'))] = Opinion.get('polarity')

        ap_list = list(dict.fromkeys(ap_list))
        obj['ap'] = ap_list
        list_obj.append(obj)
    return list_obj
