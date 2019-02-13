import numpy as np
import measures

# probs = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.9, 0.1, 0]])
# preds = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1]])
# truth = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]])
#
# ave_prec = measures.average_precision(probs, truth)
#
# print(ave_prec)
#
# rankloss = measures.ranking_loss(probs, truth)
# print(rankloss)
#
# ex_acc = measures.example_based_accuracy_instances(preds, truth)
# print(ex_acc)
#
# ex_f1 = measures.example_based_f1_instances(preds, truth)
# print(ex_f1)
#
# macro_f1 = measures.macro_f1(preds, truth)
# print(macro_f1)
#
# micro_f1 = measures.micro_f1(preds, truth)
# print(micro_f1)

a = []
a.append([1, 2])
a.append([3, 4])

for i in range(10):
    print(a)

    if i >= 5:
        a = []
        a.append([9, 10])
        a.append([10, 11])
