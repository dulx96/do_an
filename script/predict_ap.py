import individual_ap_classifier as ap

while True:
    text = input("Nhap text: \n")
    print(ap.predict_outside([text]))
