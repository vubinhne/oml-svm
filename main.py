from skmultilearn.dataset import load_from_arff
from sklearn import linear_model
import numpy as np
import measures

dataset_name = 'ENRON-F'
path_to_arff_file = '/Users/AnhVu/Study/Machine_learning/Research/MEKA/DATA_FOR_MULAN/{}.arff'.format(dataset_name)
print(path_to_arff_file)
label_count = 53

label_location = "end"

arff_file_is_sparse = False

X, y, feature_names, label_names = load_from_arff(
    path_to_arff_file,
    label_count=label_count,
    label_location=label_location,
    load_sparse=arff_file_is_sparse,
    return_attribute_definitions=True
)

n_instances = X.shape[0]
n_features = X.shape[1]
n_labels = y.shape[1]

classifiers = []

for j in range(n_labels):
    classifier = linear_model.SGDClassifier(loss='hinge', tol=1e-1, max_iter=1)
    classifier.partial_fit(X[0, :], [y[0, j]], classes=[0, 1])
    classifiers.append(classifier)

predictions = np.zeros((n_instances - 1, n_labels))
truth = np.array(y[1:, :].todense())

# Start online learning
for i in range(1, n_instances):
    if i % 100 == 0:
        print(i)
    for j in range(n_labels):
        # make prediction
        y_pred = classifiers[j].predict(X[i, :])
        predictions[i - 1, j] = y_pred

        # update classifier
        classifiers[j].partial_fit(X[i, :], [y[i, j]])

np.savetxt('{}_predictions.csv'.format(dataset_name), predictions, delimiter=",", fmt='%d')
np.savetxt('{}_truth.csv'.format(dataset_name), truth, delimiter=",", fmt='%d')

# Calculate measures based on 'predictions' and grouth-true labels 'truth'
# average precision and ranking loss require probabilities as a param, but SGD
# with hinge loss is not a probabilistic model
ave_prec = measures.average_precision(predictions, truth)
print('Average Precision = {}'.format(ave_prec))

rankloss = measures.ranking_loss(predictions, truth)
print('Ranking Loss = {}'.format(rankloss))

ex_acc = measures.example_based_accuracy_instances(predictions, truth)
print('Example-based Accuracy = {}'.format(ex_acc))

ex_f1 = measures.example_based_f1_instances(predictions, truth)
print('Example-based F1 = {}'.format(ex_f1))

macro_f1 = measures.macro_f1(predictions, truth)
print('Macro F1 = {}'.format(macro_f1))

micro_f1 = measures.micro_f1(predictions, truth)
print('Micro F1 = {}'.format(micro_f1))
