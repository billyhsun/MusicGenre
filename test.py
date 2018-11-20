def predict_genre(filename):
    pred = 1
    confidence = 0.674
    prediction = ""
    if pred == 0:
        prediction = "pop"
    if pred == 1:
        prediction = "rap"
    if pred == 2:
        prediction = "rock"
    print("This is a "+prediction+" song with a confidence of "+str(confidence))
