from sklearn.metrics import confusion_matrix, classification_report

def print_metrics(true, pred, labels):
  print(confusion_matrix(true, pred, labels=labels))