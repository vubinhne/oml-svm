import numpy as np
import measures

preds = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.9, 0.1, 0]])
truth = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]])

ave_prec = measures.average_precision(preds, truth)

print(ave_prec)

rankloss = measures.ranking_loss(preds, truth)
print(rankloss)
