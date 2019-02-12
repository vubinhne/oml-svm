import numpy as np


def ranking_loss(p_outputs, p_test_target):
    # Computing the average precision
    # Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class
    #          is stored in Outputs(i,j)
    # test_target: the actual labels of the test instances, if the ith instance belong to the
    #          jth class, test_target(i, j) = 1, otherwise test_target(i, j) = 0

    # Implemented by Vu: convert to the right format:
    outputs = np.copy(p_outputs)
    test_target = np.copy(p_test_target)
    outputs = np.transpose(outputs)
    test_target[test_target.round() == 0] = -1
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
        label_size[i] = np.sum(temp.round() == np.ones(num_class, ))
        for j in range(num_class):
            if round(temp[j]) == 1:
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


def average_precision(p_outputs, p_test_target):
    # Computing the average precision
    # Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class
    #          is stored in Outputs(i,j)
    # test_target: the actual labels of the test instances, if the ith instance belong to the
    #          jth class, test_target(i, j) = 1, otherwise test_target(i, j) = 0

    # Implemented by Vu: convert to the right format:
    outputs = np.copy(p_outputs)
    test_target = np.copy(p_test_target)
    outputs = np.transpose(outputs)
    test_target[test_target.round() == 0] = -1
    test_target = np.transpose(test_target)
    # Done

    [num_class, num_instance] = outputs.shape
    temp_outputs = []
    temp_test_target = []
    for i in range(num_instance):
        temp = test_target[:, i]
        # print('temp.shape = {}'.format(temp.shape))
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
        label_size[i] = np.sum(temp.round() == np.ones(num_class, ))
        for j in range(num_class):
            if round(temp[j]) == 1:
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
            locs = np.where(index == round(label[i][m]))
            loc = locs[0][-1]
            indicator[loc] = 1

        summary = 0
        for m in range(int(label_size[i])):
            locs = np.where(index == round(label[i][m]))
            loc = locs[0][-1]
            temp = np.sum(indicator[loc:]) / (num_class - loc + 1)
            summary = summary + np.sum(indicator[loc:]) / (num_class - loc)

        aveprec = aveprec + summary / label_size[i]

    return aveprec / num_instance


def example_based_accuracy_instance(ypred, ytruth):
    # P_ACCURACY Jaccard Index for one instance
    # often simply called multi-label 'accuracy'. Multi-label only.
    # shape: (n_labels,)

    L = ytruth.shape[0]
    set_union = 0
    set_inter = 0

    for j in range(L):
        if round(ytruth[j]) == 1 or round(ypred[j]) == 1:
            set_union = set_union + 1
        if round(ytruth[j]) == 1 and round(ypred[j]) == 1:
            set_inter = set_inter + 1

    # = 1 if both sets are empty else = intersection/union
    if set_union > 0:
        return set_inter / set_union
    else:
        return 1


def example_based_accuracy_instances(YPred, YTruth):
    # P_ACCURACY Jaccard Index for one instances
    # often simply called multi-label 'accuracy'. Multi-label only.
    # shape: (n_instances, n_labels)

    N = YTruth.shape[0]
    accuracy = 0
    for i in range(N):
        accuracy += example_based_accuracy_instance(YPred[i, :], YTruth[i, :])

    return accuracy / N


def cal_f_measure(tp, fp, fn, beta):
    if round(tp + fp + fn) == 0:
        return 1
    else:
        beta2 = beta * beta
        f_measure = ((beta2 + 1) * tp) / ((beta2 + 1) * tp + beta2 * fn + fp)
        return f_measure


def example_based_f1_instance(ypred, ytruth):
    # shape of ypred and ytruth: (n_labels,)
    L = ytruth.shape[0]
    tp = 0
    fp = 0
    fn = 0

    for j in range(L):
        if round(ytruth[j]) == 1 and round(ypred[j]) == 1:
            tp = tp + 1
        if round(ytruth[j]) == 0 and round(ypred[j]) == 1:
            fp = fp + 1
        if round(ytruth[j]) == 1 and round(ypred[j]) == 0:
            fn = fn + 1

    f1 = cal_f_measure(tp, fp, fn, 1)
    return f1


def example_based_f1_instances(YPred, YTruth):
    N = YTruth.shape[0]
    f1 = 0
    for i in range(N):
        f1 += example_based_f1_instance(YPred[i, :], YTruth[i, :])

    return f1 / N


def macro_f1(YPred, YTruth):
    [N, L] = YTruth.shape
    true_positives = np.zeros((L,))
    false_negatives = np.zeros((L,))
    false_positives = np.zeros((L,))
    true_negatives = np.zeros((L,))

    for i in range(N):
        for j in range(L):
            actual = round(YTruth[i, j])
            predicted = round(YPred[i, j])

            if actual == 1:
                if predicted == 1:
                    true_positives[j] += 1
                else:
                    false_negatives[j] += 1
            else:
                if predicted == 1:
                    false_positives[j] += 1
                else:
                    true_negatives[j] += 1

    res = 0
    for j in range(L):
        f1 = cal_f_measure(true_positives[j], false_positives[j], false_negatives[j], 1)
        res += f1

    res = res / L

    return res


def micro_f1(YPred, YTruth):
    [N, L] = YTruth.shape
    true_positives = np.zeros((L,))
    false_negatives = np.zeros((L,))
    false_positives = np.zeros((L,))
    true_negatives = np.zeros((L,))

    for i in range(N):
        for j in range(L):
            actual = round(YTruth[i, j])
            predicted = round(YPred[i, j])

            if actual == 1:
                if predicted == 1:
                    true_positives[j] += 1
                else:
                    false_negatives[j] += 1
            else:
                if predicted == 1:
                    false_positives[j] += 1
                else:
                    true_negatives[j] += 1

    tp = 0
    fp = 0
    fn = 0
    for j in range(L):
        tp += true_positives[j]
        fp += false_positives[j]
        fn += false_negatives[j]

    res = cal_f_measure(tp, fp, fn, 1)
    return res
