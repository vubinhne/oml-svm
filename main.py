from skmultilearn.dataset import load_from_arff
from sklearn import linear_model
import numpy as np
import measures
import sys
import scipy.io

# file_list = ['ENRON-F', 'OHSUMED-F', 'SLASHDOT-F', 'tmc2007-500-F', 'IMDB-F']
#     'IMDB-drift', 'SynRTG-drift',...
#     'IMDB-break', 'SynRTG-break',...
#     'ENRON-drift', 'OHSUMED-drift', 'SLASHDOT-drift', 'tmc2007-500-drift',...
#     'ENRON-break', 'OHSUMED-break', 'SLASHDOT-break', 'tmc2007-500-break'};

file_list = ['IMDB-drift', 'SynRTG-drift', 'IMDB-break', 'SynRTG-break', 'ENRON-drift', 'OHSUMED-drift',
             'SLASHDOT-drift', 'tmc2007-500-drift', 'ENRON-break', 'OHSUMED-break', 'SLASHDOT-break',
             'tmc2007-500-break']

for dataset_name in file_list:
    print(dataset_name)
    mat = scipy.io.loadmat('/Users/AnhVu/Study/Machine_learning/DenClus/MultiLabel/{}.mat'.format(dataset_name))
    D = mat['D']
    L = mat['L'][0][0]
    y = D[:, :L]
    X = D[:, L:]

    print(X.shape)
    print(y.shape)

    n_instances = X.shape[0]
    n_features = X.shape[1]
    n_labels = y.shape[1]

    classifiers = []

    for j in range(n_labels):
        classifier = linear_model.SGDClassifier(loss='modified_huber', tol=1e-1, max_iter=1)
        classifier.partial_fit([X[0, :]], [y[0, j]], classes=[0, 1])
        classifiers.append(classifier)

    predictions = np.zeros((n_instances - 1, n_labels))
    probabilities = np.zeros((n_instances - 1, n_labels))

    truth = y[1:, :]
    # truth = np.array(y[1:, :].todense())

    # Start online learning
    for i in range(1, n_instances):

        cur_probs = np.zeros(n_labels)

        for j in range(n_labels):
            # make prediction
            y_pred = classifiers[j].predict([X[i, :]])
            cur_probs[j] = np.max(classifiers[j].predict_proba([X[i, :]]))

            predictions[i - 1, j] = y_pred

            # update classifier
            classifiers[j].partial_fit([X[i, :]], [y[i, j]])

        if i % 100 == 0:
            print(i)

    # np.savetxt('{}_predictions.csv'.format(dataset_name), predictions, delimiter=",", fmt='%d')
    # np.savetxt('{}_truth.csv'.format(dataset_name), truth, delimiter=",", fmt='%d')

    # Calculate measures based on 'predictions' and grouth-true labels 'truth'
    # average precision and ranking loss require probabilities as a param, but SGD
    # with hinge loss is not a probabilistic model
    ave_prec = measures.average_precision(probabilities, truth)
    print('Average Precision = {}'.format(ave_prec))

    rankloss = measures.ranking_loss(probabilities, truth)
    print('Ranking Loss = {}'.format(rankloss))

    ex_acc = measures.example_based_accuracy_instances(predictions, truth)
    print('Example-based Accuracy = {}'.format(ex_acc))

    ex_f1 = measures.example_based_f1_instances(predictions, truth)
    print('Example-based F1 = {}'.format(ex_f1))

    macro_f1 = measures.macro_f1(predictions, truth)
    print('Macro F1 = {}'.format(macro_f1))

    micro_f1 = measures.micro_f1(predictions, truth)
    print('Micro F1 = {}'.format(micro_f1))

    f = open("result/{}.csv".format(dataset_name), "w+")
    f.write('ExF1, ExAc, MicF1, MacF1, AvePrec, Rankloss\n')
    f.write('{},{},{},{},{},{}\n'.format(ex_f1, ex_acc, micro_f1, macro_f1, ave_prec, rankloss))
    f.close()
