from sklearn.metrics import classification_report, confusion_matrix

def evaluate(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    return confusion_matrix(y_true, y_pred)