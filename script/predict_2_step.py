import individual_ap_classifier as aspect_category
import individual_sentiment_with_ap_multi_classifier as sentiment

while True:
    inputText = input('nhap text: !!! \n')
    predictedAP = aspect_category.predict_outside([inputText])
    print(predictedAP)
    for ap in predictedAP:
        print(ap)
        predicted_sentiment = sentiment.predict_outside([inputText], ap)
        print(predicted_sentiment)
