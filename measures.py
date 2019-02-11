import numpy as np


def ranking_loss(outputs, test_target):
    # Computing the average precision
    # Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class
    #          is stored in Outputs(i,j)
    # test_target: the actual labels of the test instances, if the ith instance belong to the
    #          jth class, test_target(i, j) = 1, otherwise test_target(i, j) = 0

    # Implemented by Vu: convert to the right format:
    outputs = np.transpose(outputs)
    test_target[test_target == 0] = -1
    test_target = np.transpose(test_target)
    # Done

    [num_class, num_instance] = outputs.shape
    temp_outputs = []
    temp_test_target = []
    for i in range(num_instance):
        temp = test_target[:, i]
        if np.sum(temp) != num_class and np.sum(temp) != -num_class:
            temp_outputs.append(outputs[:, i])
            temp_test_target.append(temp)

    outputs = np.transpose(np.array(temp_outputs))
    test_target = np.transpose(np.array(temp_test_target))
    [num_class, num_instance] = outputs.shape

    label = [None] * num_instance
    label = [[] if v is None else v for v in label]
    not_label = [None] * num_instance
    not_label = [[] if v is None else v for v in not_label]
    label_size = np.zeros((num_instance,))
    for i in range(num_instance):
        temp = test_target[:, i]
        label_size[i] = np.sum(temp == np.ones(num_class, ))
        for j in range(num_class):
            if temp[j] == 1:
                label[i].append(j)
            else:
                not_label[i].append(j)

    rankloss = 0
    for i in range(num_instance):
        temp = 0
        for m in range(int(label_size[i])):
            for n in range(int(num_class - label_size[i])):
                if outputs[label[i][m], i] <= outputs[not_label[i][n], i]:
                    temp = temp + 1
        rankloss = rankloss + temp / (label_size[i] * (num_class - label_size[i]))

    return rankloss / num_instance


def average_precision(outputs, test_target):
    # Computing the average precision
    # Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class
    #          is stored in Outputs(i,j)
    # test_target: the actual labels of the test instances, if the ith instance belong to the
    #          jth class, test_target(i, j) = 1, otherwise test_target(i, j) = 0

    # Implemented by Vu: convert to the right format:
    outputs = np.transpose(outputs)
    test_target[test_target == 0] = -1
    test_target = np.transpose(test_target)
    # Done

    [num_class, num_instance] = outputs.shape
    temp_outputs = []
    temp_test_target = []
    for i in range(num_instance):
        temp = test_target[:, i]
        if np.sum(temp) != num_class and np.sum(temp) != -num_class:
            temp_outputs.append(outputs[:, i])
            temp_test_target.append(temp)

    outputs = np.transpose(np.array(temp_outputs))
    test_target = np.transpose(np.array(temp_test_target))
    [num_class, num_instance] = outputs.shape

    label = [None] * num_instance
    label = [[] if v is None else v for v in label]
    not_label = [None] * num_instance
    not_label = [[] if v is None else v for v in not_label]
    label_size = np.zeros((num_instance,))
    for i in range(num_instance):
        temp = test_target[:, i]
        label_size[i] = np.sum(temp == np.ones(num_class, ))
        for j in range(num_class):
            if temp[j] == 1:
                label[i].append(j)
            else:
                not_label[i].append(j)

    aveprec = 0
    for i in range(num_instance):
        temp = outputs[:, i]
        index = np.argsort(temp)  # ascending
        tempvalue = temp[index]
        indicator = np.zeros(num_class, )
        for m in range(int(label_size[i])):
            locs = np.where(index == label[i][m])
            loc = locs[0][-1]
            indicator[loc] = 1

        summary = 0
        for m in range(int(label_size[i])):
            locs = np.where(index == label[i][m])
            loc = locs[0][-1]
            temp = np.sum(indicator[loc:]) / (num_class - loc + 1)
            summary = summary + np.sum(indicator[loc:]) / (num_class - loc)

        aveprec = aveprec + summary / label_size[i]

    return aveprec / num_instance

