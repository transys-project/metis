import json
import copy
from sklearn.tree._tree import TREE_LEAF
from math import sqrt
import numpy as np
import math

def rules(clf, features, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # 叶子节点
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        node['name'] = 'leaf'
    else:
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} <= {}'.format(feature, threshold)
        left_index = clf.tree_.children_right[node_index]
        right_index = clf.tree_.children_left[node_index]
        node['children'] = [rules(clf, features, right_index),
                            rules(clf, features, left_index)]
    return node


def tree_node_count(model,count):#|Tt|
    if "children" not in model:
        return 1
    children = model["children"]
    for child in children:
        count += tree_node_count(child, 0)
    return count


def Rt_compute(Rt_compute):
    # 1. Accuracy
    # Rt=(sum(model['value'])-max(model['value']))
    # return Rt
    # 2. Gini loss
    sq_count = 0.0
    samples = np.sum(Rt_compute['value'])
    for k in Rt_compute['value']:
        sq_count += k * k
    gini = 1.0 - sq_count / (samples * samples)
    return samples * gini


def RTt_compute(model, leaves_error_count):
    if "children"not in model:
        return Rt_compute(model)
    children = model["children"]
    for child in children:
        leaves_error_count += RTt_compute(child, 0)
    return leaves_error_count


def gt_compute(json_tree):
    return (Rt_compute(json_tree)-RTt_compute(json_tree, 0)*1.0) / (tree_node_count(json_tree, 0)-1)


def T1_create(model, gt_list, prune_parts, prune_gt_index):
    if 'children' not in model:
        return
    else:
        gt_list.append(gt_compute(model))
        prune_parts.append(model)
        children = model["children"]
        if len(prune_parts) == prune_gt_index+1:
            del model["children"]
        for child in children:
            T1_create(child, gt_list, prune_parts, prune_gt_index)


def gt_with_tree(json_tree, gt_list, prune_parts, path, path_list):
    if 'children' not in json_tree:
        return
    else:
        gt_list.append(gt_compute(json_tree))
        path_list.append(path)
        prune_parts.append(json_tree)
        children = json_tree["children"]
        for i in range(len(children)):
            gt_with_tree(children[i], gt_list, prune_parts, path+[i], path_list)


def prune_sklearn_model(sklearn_model, index, json_model):
    if "children" not in json_model:
        # Modify values of leaf nodes
        for i in range(len(json_model['value'])):
            sklearn_model.value[index][0][i] = json_model['value'][i]
        sklearn_model.children_left[index] = TREE_LEAF
        sklearn_model.children_right[index] = TREE_LEAF
    else:
        prune_sklearn_model(sklearn_model,sklearn_model.children_left[index],json_model["children"][0])
        prune_sklearn_model(sklearn_model,sklearn_model.children_right[index],json_model["children"][1])


def model_gtmin_Tt(tree, json_tree):    # T0->T1
    gt_list = []
    prune_parts = []
    path = [1]
    path_list = []
    gt_with_tree(json_tree, gt_list, prune_parts, path, path_list)
    alpha = min(gt_list)
    prune_gt_index = gt_list.index(alpha)
    # Delete child by child-path
    T1 = copy.deepcopy(json_tree)
    temp_tree = T1
    for i in path_list[prune_gt_index][1:]:
        temp_tree = temp_tree['children'][i]
    del temp_tree["children"]

    # Old version
    # T1 = copy.deepcopy(json_tree)
    # gt_list = []
    # prune_parts = []
    # T1_create(T1, gt_list, prune_parts, prune_gt_index)
    return tree, T1, alpha, path_list[prune_gt_index]


def path2rank(path):
    rank = 0
    for i in path:
        rank *= 2
        rank += i
    return int(rank)


def rank2path(rank):
    rank = int(rank)
    path = []
    while rank >= 1:
        path.insert(0, rank % 2)
        rank = math.floor(rank / 2)
    return path


def get_rebuf(state):
    rebuf = 0.0
    return rebuf


def check_reward(json_tree, path, x_train, nodeRankDict, parameters, classes, copies, env):
    # 1. Get values and test cases of the chosen node
    temp_tree = copy.deepcopy(json_tree)
    node = temp_tree
    for i in path[1:]:
        node = node['children'][i]
    values = node['value']
    rank = path2rank(path)
    if rank not in nodeRankDict:
        indexes = []
        rankQueue = [rank]
        while True:
            if len(rankQueue) == 0:
                break
            nowRank = rankQueue.pop(0)
            nowRankLeft = nowRank * 2
            nowRankRight = nowRank * 2 + 1
            if nowRankLeft in nodeRankDict:
                indexes += nodeRankDict[nowRankLeft]
                del nodeRankDict[nowRankLeft]
            else:
                rankQueue.append(nowRankLeft)
            if nowRankRight in nodeRankDict:
                indexes += nodeRankDict[nowRankRight]
                del nodeRankDict[nowRankRight]
            else:
                rankQueue.append(nowRankRight)
        nodeRankDict[rank] = indexes
    indexes = nodeRankDict[rank]
    # 2. Traverse all bit_rate values, compare reward value by passing states to fixed_env
    maxValue = max(values) + 9000000
    rewards = []
    for i in range(len(values)):
        temp = values[i]
        node['value'][i] = maxValue
        reward = 0
        bit_rate = int(classes[i])
        for index in indexes:
            idx_copy = copies[index]
            state = x_train[index]
            trace_idx = int(idx_copy[0])
            mahimahi_ptr = int(idx_copy[1])
            video_chunk_counter = int(idx_copy[2])
            buffer_size = idx_copy[3]

            cooked_time = env.all_cooked_time[trace_idx]
            cooked_bw = env.all_cooked_bw[trace_idx]
            video_chunk_size = env.video_size[bit_rate][video_chunk_counter]
            last_mahimahi_time = cooked_time[mahimahi_ptr - 1]
            delay = 0.0
            video_chunk_counter_sent = 0
            while True:  # download video chunk over mahimahi
                throughput = cooked_bw[mahimahi_ptr] * parameters['B_IN_MB'] / parameters['BITS_IN_BYTE']
                duration = cooked_time[mahimahi_ptr] - last_mahimahi_time
                packet_payload = throughput * duration * parameters['PACKET_PAYLOAD_PORTION']
                if video_chunk_counter_sent + packet_payload > video_chunk_size:
                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                      throughput / parameters['PACKET_PAYLOAD_PORTION']
                    delay += fractional_time
                    last_mahimahi_time += fractional_time
                    assert (last_mahimahi_time <= cooked_time[mahimahi_ptr])
                    break
                video_chunk_counter_sent += packet_payload
                delay += duration
                last_mahimahi_time = cooked_time[mahimahi_ptr]
                mahimahi_ptr += 1
                if mahimahi_ptr >= len(cooked_bw):
                    mahimahi_ptr = 1
                    last_mahimahi_time = 0
            delay *= parameters['MILLISECONDS_IN_SECOND']
            delay += parameters['LINK_RTT']
            rebuf = np.maximum(delay - buffer_size, 0.0)
            last = float(np.max(parameters['VIDEO_BIT_RATE'])) * state[0]
            reward = parameters['VIDEO_BIT_RATE'][bit_rate] / parameters['M_IN_K'] - \
                     4.3 * rebuf - parameters['SMOOTH_PENALTY'] * np.abs(parameters['VIDEO_BIT_RATE'][bit_rate] - last) \
                     / 1000.0

        rewards.append(reward)
        node['value'][i] = temp
    # 3. Get the chosen bit_rate, modify json_tree
    index = rewards.index(max(rewards))
    node['value'][index] = maxValue
    return temp_tree


def candidate(clf, json_model, alpha_list, tree_list, max_leaf_nodes, TEST_REWARD,\
              x_train, nodeRankDict, parameters, classes, copies, env):
    flag = True
    alpha = 0
    tree_name = 0
    tree = copy.deepcopy(clf)
    json_tree = copy.deepcopy(json_model)

    while flag:
        leaf_num = str(json_tree).count('name')
        if max_leaf_nodes is None or leaf_num < max_leaf_nodes * 2:
            alpha_list.append(alpha)
            tree_list.append(json_tree)

        tree_name = tree_name + 1
        tree, json_tree, alpha, path = model_gtmin_Tt(tree, json_tree)

        if TEST_REWARD:
            json_tree = check_reward(json_tree, path, x_train, nodeRankDict, parameters, classes, copies, env)

        if "children" not in json_tree:
            tree_list.append(copy.deepcopy(json_tree))
            alpha_list.append(alpha)
            flag = False

    return alpha_list, tree_list


def classify(json_model, test_data, features, classes, path):
    if "children" not in json_model:
        return json_model["value"], path  # 到达叶子节点，完成测试

    bestfeature = json_model["name"].split("<=")[0].strip()
    threshold = float(json_model["name"].split(bestfeature+" <= ")[1].strip())
    test_best_feature_value = test_data[features.index(bestfeature)]
    if float(test_best_feature_value) <= threshold:
        child = json_model["children"][0]
        result, path = classify(child, test_data, features, classes, path+[0])
    else:
        child = json_model["children"][1]
        result, path = classify(child, test_data, features, classes, path+[1])
    return result, path


def predict(json_model, test_item, features, classes):
    leaf_value, path = classify(json_model, test_item, features, classes, [1])
    class_names_index = leaf_value.index(max(leaf_value))
    return classes[class_names_index], class_names_index, path


def reward_compute(json_model, x_test, copies_test, parameters, env, features, classes):
    rewards = 0.0
    for index in range(len(x_test)):
        idx_copy = copies_test[index]
        state = x_test[index]
        bit_rate, class_name_index, path = predict(json_model, state, features, classes)
        bit_rate = int(bit_rate)
        trace_idx = int(idx_copy[0])
        mahimahi_ptr = int(idx_copy[1])
        video_chunk_counter = int(idx_copy[2])
        buffer_size = idx_copy[3]

        cooked_time = env.all_cooked_time[trace_idx]
        cooked_bw = env.all_cooked_bw[trace_idx]
        video_chunk_size = env.video_size[bit_rate][video_chunk_counter]
        last_mahimahi_time = cooked_time[mahimahi_ptr - 1]
        delay = 0.0
        video_chunk_counter_sent = 0
        while True:  # download video chunk over mahimahi
            throughput = cooked_bw[mahimahi_ptr] * parameters['B_IN_MB'] / parameters['BITS_IN_BYTE']
            duration = cooked_time[mahimahi_ptr] - last_mahimahi_time
            packet_payload = throughput * duration * parameters['PACKET_PAYLOAD_PORTION']
            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / parameters['PACKET_PAYLOAD_PORTION']
                delay += fractional_time
                last_mahimahi_time += fractional_time
                assert (last_mahimahi_time <= cooked_time[mahimahi_ptr])
                break
            video_chunk_counter_sent += packet_payload
            delay += duration
            last_mahimahi_time = cooked_time[mahimahi_ptr]
            mahimahi_ptr += 1
            if mahimahi_ptr >= len(cooked_bw):
                mahimahi_ptr = 1
                last_mahimahi_time = 0
        delay *= parameters['MILLISECONDS_IN_SECOND']
        delay += parameters['LINK_RTT']
        rebuf = np.maximum(delay - buffer_size, 0.0)
        last = float(np.max(parameters['VIDEO_BIT_RATE'])) * state[0]
        reward = parameters['VIDEO_BIT_RATE'][bit_rate] / parameters['M_IN_K'] - \
                 4.3 * rebuf - parameters['SMOOTH_PENALTY'] * np.abs(parameters['VIDEO_BIT_RATE'][bit_rate] - last) \
                 / 1000.0
        rewards += reward
    return rewards


def validate_reward(TreeSets, x_test, sklearn_model, copies_test, parameters, env, features, classes):
    reward_list = []
    for index, item in enumerate(TreeSets):
        reward = reward_compute(item, x_test, copies_test, parameters, env, features, classes)
        reward_list.append(reward)
    index = reward_list.index(max(reward_list))
    best_json_tree = TreeSets[index]
    best_sklearn_model = copy.deepcopy(sklearn_model)
    prune_sklearn_model(best_sklearn_model.tree_, 0, best_json_tree)
    prune_node_count = str(best_json_tree).count('name')
    print('During pruning, best tree is', index)
    print('Node Count', prune_node_count)
    return best_sklearn_model

def precision_compute(json_model, X_test, y, features, classes):
    count_right = 0.0
    X_test = np.array(X_test).tolist()
    for index, item in enumerate(X_test):
        result, class_name_index, path = predict(json_model, item, features, classes)
        if result == str(y[index]):
            count_right += 1
    return count_right/len(X_test)


def validate(TreeSets, alpha_list, x_test, y_test, sklearn_model, b_SE, features, classes, copies_test):
    precision_list = []
    for index, item in enumerate(TreeSets):
        Ti_precision = precision_compute(item, x_test, y_test, features, classes)
        precision_list.append(Ti_precision)
    if b_SE == False:
        pruned_precision = max(precision_list)
        index = precision_list.index(pruned_precision)
        best_alpha = alpha_list[index]
        Best_tree = TreeSets[index]

        best_sklearn_model = copy.deepcopy(sklearn_model)
        prune_sklearn_model(best_sklearn_model.tree_, 0, Best_tree)
        return Best_tree, best_alpha, pruned_precision, precision_list[0]
    else:
        error_rate_list = [1 - item for item in precision_list]
        lowest_error_rate = min(error_rate_list)
        SE = sqrt(lowest_error_rate * (1 - lowest_error_rate) / len(y_test))
        criterion_1_SE = lowest_error_rate + SE

        index_error_rate = 0
        for index, item in enumerate(
                error_rate_list):  # search from from the end ,because the error_rate_list is not monotory.

            if error_rate_list[len(error_rate_list) - 1 - index] < criterion_1_SE:
                index_error_rate = len(error_rate_list) - 1 - index
                break

        # if index_error_rate-1>=0:
        #     index_error_rate=index_error_rate-1
        # else:
        #     pass#becasuse the list may only have one item.

        # Add for test
        pruned_precision = precision_list[
            index_error_rate]  # here's right,because the precision list is corresponding to the error_rate_list.
        best_alpha = alpha_list[index_error_rate]
        Best_tree = TreeSets[index_error_rate]
        best_sklearn_model = copy.deepcopy(sklearn_model)
        prune_sklearn_model(best_sklearn_model.tree_, 0, Best_tree)
        prune_node_count = str(Best_tree).count('name')
        print('During pruning, best tree is', index_error_rate)
        print('Node Count', prune_node_count)
        print('best_alpha', best_alpha, 'pruned_precision', pruned_precision, 'unpruned_precision', precision_list[0])
        return best_sklearn_model, Best_tree


def prune(tree, features, classes, x_test, y_test, max_leaf_nodes, x_train, copies_train, copies_test, parameters, env):
    json_model = rules(tree, features)
    # with open('structure.json', 'w') as f:
    #     f.write(json.dumps(json_model))

    # path -> index -> states
    nodeRankDict = {}  # nodeRank -> states-indexes
    for index, item in enumerate(x_train):
        result, class_name_index, path = predict(json_model, item, features, classes)
        rank = path2rank(path)
        if rank not in nodeRankDict:
            nodeRankDict[rank] = []
        nodeRankDict[rank].append(index)

    alpha_list = []
    tree_list = []

    REWARD = True
    alpha_list, tree_list = candidate(copy.deepcopy(tree), copy.deepcopy(json_model), alpha_list, tree_list,\
                                      max_leaf_nodes, REWARD, x_train, nodeRankDict, parameters, classes, copies_train, env)
    # best_tree, best_json_tree = validate(tree_list, alpha_list, x_test, y_test, copy.deepcopy(tree), \
    #                                             True, features, classes, copies_test)
    best_tree = validate_reward(tree_list, x_test, copy.deepcopy(tree), copies_test, parameters, env, features, classes)
    return best_tree
