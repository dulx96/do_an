import individual_sentiment_with_ap_multi_classifier as sentiment

while True:
    text = input('Nhap text: \n')
    ap = input('Nhap ap: \n')
    sentiment_predict = sentiment.predict_outside([text],ap)
    print(sentiment_predict)
