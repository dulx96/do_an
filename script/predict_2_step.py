import individual_ap_classifier as ap
import individual_sentiment_with_ap_multi_classifier as sentiment

apList = ap.aspect_category_list
while True:
    inputText = input('nhap text: !!! \n')
    predictedAP = ap.predict_outside([inputText])
    print(predictedAP)
    for ap_predicted in predictedAP:
        if round(ap_predicted['H']) >= 50:
            print(ap_predicted['ap'])
            predicted_sentiment = sentiment.predict_outside([inputText], ap_predicted['ap'])
            print(predicted_sentiment)
