
def calculate_metrics(predictions, labels, adjust):
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  if adjust:
    preds = adjust_predictions(predictions, labels)
  for i in range(len(predictions)):
    if preds[i] == 1 and labels[i] == 1:
      tp = tp + 1
    elif preds[i] == 0 and labels[i] == 0:
      tn = tn + 1
    elif preds[i] == 1 and labels[i] == 0:
      fp = fp + 1
    else: # preds[i] == 0 and labels[i] == 1:
      fn = fn + 1

  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  f1 = 2 * precision * recall / (precision + recall)
  return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'presision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1}

#just like onmianomaly, no delta. If we hit anuthing in anomaly interval, whole anomaly segment is correctly identified
#-----------------------
#1|0|1|1|1|0|0|0|1|1|1|1  Labels
#-----------------------
#0|0|0|1|1|0|0|0|0|0|1|0  Predictions
#-----------------------
#0|0|1|1|1|0|0|0|1|1|1|1  Adjusted
#-----------------------
def adjust_predictions(predictions, labels):
  adjustment_started = False

  for i in range(len(predictions)):
    if labels[i] == 1:
      if predictions[i] == 1:
        if not adjustment_started:
          adjustment_started = True
          for j in range(i, 0, -1):
            if labels[j] == 1:
              predictions[j] = 1
            else:
              break
    else:
      adjustment_started = False

    if adjustment_started:
      predictions[i] = 1
      
  return predictions

if __name__ == '__main__':
  labels = [1,0,1,1,1,0,0,0,1,1,1,1]
  predictions = [0,0,0,1,1,0,0,0,0,0,1,0]

  print(adjust_predictions(predictions, labels))
