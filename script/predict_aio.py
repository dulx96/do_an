import aio_aspect_sentiment as aio

while True:
    text = input('Nhap text: \n')
    print(aio.predict_outside([text]))
