from skmultilearn.dataset import load_from_arff
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn import linear_model
import numpy as np

path_to_arff_file = '/Users/AnhVu/Study/Machine_learning/Research/MEKA/DATA_FOR_MULAN/ENRON-F.arff'
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
    classifier = linear_model.SGDClassifier(tol=1e-3)
    classifier.partial_fit(X[0, :], [y[0, j]])
    classifiers.append(classifier)

predictions = np.zeros((n_instances - 1, n_labels))

# Start online learning
for i in range(1, n_instances):
    for j in range(n_labels):
        # make prediction
        y_pred = classifiers[j].predict(X[i, :])
        predictions[i - 1, j] = y_pred

        # update classifier
        classifiers[j].partial_fit(X[i, :], [y[i, j]])

# Calculate measures based on 'predictions' and grouth-true labels 'y'
