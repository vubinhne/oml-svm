import average_precision
import numpy as np

preds = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.9, 0.1, 0]])
truth = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]])

res = average_precision.average_precision(preds, truth)

print(res)
