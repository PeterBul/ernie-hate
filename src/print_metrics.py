from sklearn.metrics import confusion_matrix, classification_report

def print_metrics(true, pred, target_names):
  print(classification_report(true, pred, target_names=target_names, digits=8))
  print(confusion_matrix(true, pred))