"""
Utrecht University, Data Mining Assignment 1

Miguel Â. Simões Valente    6876005
Riccardo Bassani            6866840
Samuel Meyer                5648122
"""

import numpy as np


class Node:
    """
    A class representing a node.
    Each node contains attributes for its children, so the root node is equivalent to the whole tree.
    Each node also contains the split defined on it during the training process, and the observation in the
    training set used to find the best split.
    """
    def __init__(self, train_attr, train_lab):
        self.left = None                # a node is a leaf by default
        self.right = None               # a node is a leaf by default
        self.split = None               # a node is a leaf by default (not split)
        self.indentation = ""           # indentation to facilitate the nice printing
        self.train_attr = train_attr    # a (numpy) matrix of attributes*instances
        self.train_lab = train_lab      # an array of binary labels, corresponding to the attributes

    # setters
    def set_split(self, s):
        self.split = s
    def set_left(self, l):
        self.left = l
    def set_right(self, r):
        self.right = r
    def set_indentation(self, i):
        self.indentation = i

    def __str__(self):
        """
        Nice print for the tree.
        :param self: the root node of the tree to print.
        :return:     a string representing the tree. Each level indented appropriately.
        """
        if self.split is None:
            return "leaf node --> " + str(self.get_majority_label())
        return "\n" + self.indentation + "Split: " + str(self.split) + "\n" + self.indentation + "left: " + str(self.left) + "\n" + self.indentation +"right: " + str(self.right)

    def get_observation_number(self):
        """
        Return the number of observations in the node.
        :param self: the node to consider.
        :return: the number of observations in the node.
        """
        return len(self.train_lab)

    def get_impurity(self):
        """
        Return the impurity of the node, using the Gini index.
        :param self: the node to consider.
        :return: the Gini index of the node.
        """
        ones = np.count_nonzero(self.train_lab)
        tot = np.size(self.train_lab)
        return (ones/tot) * (1-(ones/tot))

    def get_majority_label(self):
        """
        Return the majority label of a node, as a binary class label (0 or 1).
        :param self: the node to consider.
        :return: the majority label od the node.
        """
        ones = np.count_nonzero(self.train_lab)
        tot = np.size(self.train_lab)
        if ones > (tot-ones):
            return 1
        return 0

    def best_split(self, minleaf, nfeat):
        """
        Compute the best split in the node and return it, together with the children created by the split.
        :param self: the node to consider.
        :param minleaf: the minimum number of observations required for a leaf node.
        :param nfeat: the number of features that should be considered for each split.
        :return: the best split, the left child, the right child.
        """
        # given train_attr and train_lab, check which attribute is best for splitting
        max_reduction = 0
        best_attr = 0
        threshold = 0
        minleaf_possible = False
        # select set of nfeat random attributes
        random_attributes = np.sort(np.random.choice(np.arange(len(self.train_attr[0])), size=nfeat, replace=False))
        # loop through the attributes
        for index in random_attributes:
            # organize values of attribute as unique ordered list
            attr_values = np.sort(np.unique(self.train_attr[:, index]))
            # loop through values for attribute to find best split based on threshold
            for attr_index, attr_value in enumerate(attr_values[:-1]):
                # calculate attribute threshold between two consecutive values
                current_threshold = (attr_values[attr_index+1] - attr_value)/2 + attr_value
                # find indices for instances given the current attribute value as threshold
                indices_l = np.arange(len(self.train_attr[:, index]))[self.train_attr[:, index] > current_threshold]
                indices_r = np.arange(len(self.train_attr[:, index]))[self.train_attr[:, index] <= current_threshold]
                # calculate gini index for each side of split
                ones_l = np.count_nonzero(self.train_lab[indices_l])
                ones_r = np.count_nonzero(self.train_lab[indices_r])
                tot_l = np.size(indices_l)
                tot_r = np.size(indices_r)
                gini_l = (ones_l / tot_l) * (1 - (ones_l / tot_l))
                gini_r = (ones_r / tot_r) * (1 - (ones_r / tot_r))

                # Check impurity reduction from split and whether the produced splits have at least minleaf instances
                current_reduction = (self.get_impurity() - (gini_l*(tot_l/len(self.train_lab)) + gini_r*(tot_r/len(self.train_lab))))
                if current_reduction > max_reduction\
                        and tot_l >= minleaf and tot_r >= minleaf:
                    minleaf_possible = True
                    best_attr = index
                    threshold = current_threshold
                    max_reduction = current_reduction

        if not minleaf_possible:
            return None, None, None

        final_indices_l = np.arange(len(self.train_attr[:, best_attr]))[self.train_attr[:, best_attr] > threshold]
        final_indices_r = np.arange(len(self.train_attr[:, best_attr]))[self.train_attr[:, best_attr] <= threshold]
        # give each node train_attr and train_lab subset with only corresponding instances
        # split given attribute and threshold values for best split
        split = Split(best_attr, threshold)
        left = Node(self.train_attr[final_indices_l], self.train_lab[final_indices_l])
        right = Node(self.train_attr[final_indices_r], self.train_lab[final_indices_r])

        return split, left, right


class Split:
    """
    A simple class representing a split:
     - left child = attr > threshold
     - right child = attr <= threshold
    """
    def __init__(self, attr, threshold):
        self.attr = attr                # the attribute the split is performed on
        self.threshold = threshold      # the threshold of the split

    def __str__(self):
        """
        Nice print for the split.
        :param self: the split to print.
        :return:     a string representing the split in the form "attribute number > threshold.
        """
        return str(self.attr) + " > " + str(self.threshold)


def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    Grow a tree given training data. Allows early stopping and features selections for random forest.
    :param x:       a data matrix (2-dimensional array) containing the (numeric) attribute values.
                    Each row of x contains the attribute values of one training example.
    :param y:       the vector (1-dimensional array) of class labels.
                    The class label is binary, with values coded as 0 and 1.
    :param nmin:    the number of observations that a node must contain at least, for it to be allowed to be split.
    :param minleaf: the minimum number of observations required for a leaf node.
    :param nfeat:   the number of features that should be considered for each split.
    :return:        the tree which has been created.
    """
    root = Node(x, y)
    if root.get_observation_number() < nmin:
        return root
    nodelist = list()
    nodelist.append(root)
    # until the nodelist is not empty
    while nodelist:
        # get a node and if it is not pure perform a split
        current_node = nodelist.pop()
        if current_node.get_impurity() > 0 and current_node.get_observation_number() >= nmin:
            split, left, right = current_node.best_split(minleaf, nfeat)
            # grow the tree only if a split has been performed
            if left is not None:
                current_node.set_split(split)
                left.set_indentation(current_node.indentation + "\t")
                current_node.set_left(left)
                right.set_indentation(current_node.indentation + "\t")
                current_node.set_right(right)
                nodelist.append(left)
                nodelist.append(right)
    return root


def tree_pred(x, tr):
    """
    Predict the class labels of an array of cases, using an already grown tree.
    :param x:   a data matrix (2-dimensional array) containing the attribute values of the cases
                for which predictions are required.
    :param tr:  a tree object needed to perform the prediction, created with the function tree_grow.
    :return:    the vector (1-dimensional array) of predicted class labels for the cases in x.
    """
    y = list()
    for obs in x:
        label = process_obs(obs, tr)
        y.append(label)
    return y


def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    """
    Grow a list of trees given training data, with a Random Forest approach: a tree is grown on each bootstrap sample.
    When all the features are considered in each split, this reduces to Bagging.
    :param x:       a data matrix (2-dimensional array) containing the (numeric) attribute values.
                    Each row of x contains the attribute values of one training example.
    :param y:       the vector (1-dimensional array) of class labels.
                    The class label is binary, with values coded as 0 and 1.
    :param nmin:    the number of observations that a node must contain at least, for it to be allowed to be split.
    :param minleaf: the minimum number of observations required for a leaf node.
    :param nfeat:   the number of features that should be considered for each split.
    :param m:       the number of bootstrap samples to be drawn.
    :return:        a list containing m trees.
    """
    tree_list = list()
    for i in range(m):
        # bootstrap sample from data
        indexes = np.random.choice(range(np.size(y)), size=np.size(y))
        x_b = x[indexes]
        y_b = y[indexes]
        # grow a tree and append it to the list to return
        tree = tree_grow(x_b, y_b, nmin, minleaf, nfeat)
        tree_list.append(tree)
    return tree_list


def tree_pred_b(x, trees):
    """
    Predict the class labels of an array of cases, using a list of trees grown with Bagging/Random Forest.
    The majority vote of the trees determines the final class label assigned to each case.
    :param x:       a data matrix (2-dimensional array) containing the attribute values of the cases
                    for which predictions are required.
    :param trees:   a list of trees object needed to perform the prediction, created with the function tree_grow_b.
    :return:        the vector (1-dimensional array) of predicted class labels for the cases in x.
    """
    # create a vector y, where y[i] contains the predicted class label for row i of x
    all_labels = list()
    for obs in x:
        rf_labels = list()
        # iterate over all trees and take the majority vote
        for tree in trees:
            rf_labels.append(process_obs(obs, tree))
        majority_vote = max(set(rf_labels), key=rf_labels.count)
        all_labels.append(majority_vote)
    return all_labels


def process_obs(obs, tr):
    """
    Process a single observation, using an already grown tree.
    :param obs: the observation to be processed.
    :param tr:  the trained tree.
    :return:    the predicted class label of the processed observation.
    """
    current_node = tr
    # continue until a leaf node is reached
    while current_node.left:
        if obs[current_node.split.attr] > current_node.split.threshold:
            current_node = current_node.left
        else:
            current_node = current_node.right
    return current_node.get_majority_label()
