import individual_ap_classifier_less_data as ap_less
import individual_ap_classifier_most_data as ap_most

thresold = 25


def predict_outside(text_array):
    predict_ap_less = ap_less.predict_outside(text_array)
    predict_ap_most = ap_most.predict_outside(text_array)
    predict_ap = predict_ap_less + predict_ap_most
    ap_list = [e['ap'] for e in predict_ap if e['H'] >= thresold]
    return ap_list
