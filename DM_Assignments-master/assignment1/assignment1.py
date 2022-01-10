"""
Utrecht University, Data Mining Assignment 1

Miguel Â. Simões Valente    6876005
Riccardo Bassani            6866840
Samuel Meyer                5648122
"""
import pickle
import pandas as pd
import numpy as np
import pathlib
import time
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# TODO  - Growing the trees takes around 4 minutes. Should we serialize them and give an option to load them?
#         Can we submit pickle files?
#       - ANSWER: No probs, the code will only be tested on small datasets
#       - offer prompt to choose between options 1/2/3? or just delete the credit and pima stuff?
#       - ANSWER: only the 4 functions + classes/helpers must be submitted, no need for code doing analysis/prints.
#       -         no need for main()!
#       - order of code? before classes or tree methods?
#       - ANSWER: it is ok to start with classes
#       -
#       - compare accuracy with other groups --> Our results are in line with Table 5 of the article (lin. regression)

# The file assignment1_6876005_6866840_5648122 contains the code to submit, cleaned and documented

# global variable to load the models
LOAD = True

# global variable to save the list of attributes
attribute_list = list()


class Node:
    """
    A class representing a node. The root node is equivalent to the whole tree.
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
        :return:     a string representing the tree. Each level indented appropriately .
        """
        # print("Total number of instances: " + str(len(self.train_attr)) + " with majority label " + str(
        #     self.get_majority_label()))
        # print("The division gives " + str(np.count_nonzero(self.train_lab)) + " ones, and " + str(len(self.train_attr) - np.count_nonzero(
        #     self.train_lab)) + " zero's")
        if self.split is None:
            return "leaf node --> " + str(self.get_majority_label())
        return "\n" + self.indentation + "Split: " + str(self.split) + " Total number of instances: " + str(len(self.train_attr)) + " The division gives: " + str(np.count_nonzero(self.train_lab)) + " ones, and: " + str(len(self.train_attr) - np.count_nonzero(
            self.train_lab)) + " zero's" +"\n" + self.indentation + "left: " + str(self.left) + "\n" + self.indentation +"right: " + str(self.right)

        # print("Total number of instances: " + str(len(current_node.train_attr)) + " with majority label " + str(
        #     current_node.get_majority_label()))
        # print("The division gives " + str(np.count_nonzero(current_node.train_lab)) + " ones, and " + str(len(current_node.train_attr) - np.count_nonzero(
        #     current_node.train_lab)) + " zero's")


    def get_observation_number(self):
        """
        Return the number of observations in the node.
        :param self: the node to consider.
        :return: the number of observations in the node.
        """
        return len(self.train_lab)
        # return self.train_lab.size()        # mine

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
        Return the majority label of a node, in the case of a binary class label (0 or 1).
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
            # print("attr", index, "\nvalues:\t", attr_values, "\n")
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
                # print("old impurity", self.get_impurity())
                # print("left: ", gini_l*(tot_l/len(self.train_lab)))
                # print("right: ", gini_r*(tot_r/len(self.train_lab)))
                # print (current_threshold, "-->", current_reduction)
                if current_reduction > max_reduction\
                        and tot_l >= minleaf and tot_r >= minleaf:
                    minleaf_possible = True
                    best_attr = index
                    threshold = current_threshold
                    max_reduction = current_reduction
                    # print("selected as far")

        if not minleaf_possible:
            # print("minleaf not possible")
            return None, None, None

        final_indices_l = np.arange(len(self.train_attr[:, best_attr]))[self.train_attr[:, best_attr] > threshold]
        final_indices_r = np.arange(len(self.train_attr[:, best_attr]))[self.train_attr[:, best_attr] <= threshold]
        # give each node train_attr and train_lab subset with only corresponding instances
        # split given attribute and threshold values for best split
        split = Split(best_attr, threshold)
        left = Node(self.train_attr[final_indices_l], self.train_lab[final_indices_l])
        right = Node(self.train_attr[final_indices_r], self.train_lab[final_indices_r])

        # Does this also work for binary attributes?
        # print("Split on ", best_attr, "with threshold ", threshold, "\n\n\n")
        return split, left, right


class Split:
    """
    A simple class representing a split:
     - left child = attr > threshold
     - right child = attr <= threshold
    """
    def __init__(self, attr, threshold):
        self.attr = attr
        self.threshold = threshold

    def __str__(self):
        return attribute_list[self.attr] + " > " + str(self.threshold)


def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    Grows a tree given training data. Allows early stopping and features selections for random forest.
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
        current_node = nodelist.pop()       # is random better? If so, remember to delete the node after getting it
        # print("impurity", current_node.get_impurity())
        # print("observation_number", current_node.get_observation_number())
        if current_node.get_impurity() > 0 and current_node.get_observation_number() >= nmin:
            split, left, right = current_node.best_split(minleaf, nfeat)
            # if left has not been set, probably due to minleaf not being possible
            # thus, don't set kid nodes and don't append to nodelist
            if left is not None:
                current_node.set_split(split)
                left.set_indentation(current_node.indentation + "\t")
                current_node.set_left(left)
                right.set_indentation(current_node.indentation + "\t")
                current_node.set_right(right)
                nodelist.append(left)
                nodelist.append(right)
        # print("Number of nodes left", len(nodelist))
    # print("Tree:", root)
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


def process_obs(obs, tr):
    """
    Process a single observation, using an already grown tree.
    :param obs: the observation to be processed.
    :param tr:  the trained tree.
    :return:    the predicted class label of the processed observation.
    """
    current_node = tr
    while current_node.left:
        if obs[current_node.split.attr] > current_node.split.threshold:
            # print("attribute", current_node.split.attr, ">", current_node.split.threshold, ": go left")
            current_node = current_node.left
        else:
            # print("attribute", current_node.split.attr, "<", current_node.split.threshold, ": go right")
            current_node = current_node.right
    return current_node.get_majority_label()


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


def print_metrics(title, prediction, true_labels):
    """
    Print the accuracy, precision and recall of a model given predictions and corresponding true labels.
    :param title:           a description of the model.
    :param prediction:      the predicted labels.
    :param true_labels:     the true labels
    :return:                none.
    """
    matrix = confusion_matrix(true_labels, prediction)
    accuracy = accuracy_score(true_labels, prediction)
    precision = precision_score(true_labels, prediction, average="binary")
    recall = recall_score(true_labels, prediction, average="binary")
    print(f"Results from: {title}")
    print(f"- Confusion Matrix: \n{matrix}")
    print(f"- Accuracy: \t{accuracy}")
    print(f"- Precision: \t{precision}")
    print(f"- Recall: \t\t{recall}\n")


def analyse_eclipse_data(load=LOAD):
    """
    Analyze the Eclipse package level data, using release 2.0 as the training set, and release 3.0 as the test set.
    Predict whether or not any post-release bugs have been reported.
    Prints accuracy, precision, recall and confusion matrix for the single tree, bagging and random forest cases.

    :param load: weather the trees must be loaded from a pickle file.
    :return:     none.
    """
    # get the data taking the path into account
    path = pathlib.Path(r"data/eclipse-metrics-packages-2.0.csv")
    df_train = pd.read_csv(path, delimiter=";")
    path = pathlib.Path(r"data/eclipse-metrics-packages-3.0.csv")
    df_test = pd.read_csv(path, delimiter=";")

    # Cleans the data. I'm sure there's a more efficient way but this does the job
    syntax_tree_features = ["AnnotationTypeDeclaration", "AnnotationTypeMemberDeclaration", "AnonymousClassDeclaration",
                            "ArrayAccess", "ArrayCreation", "ArrayInitializer", "ArrayType", "AssertStatement",
                            "Assignment", "Block", "BlockComment", "BooleanLiteral", "BreakStatement", "CastExpression",
                            "CatchClause", "CharacterLiteral", "ClassInstanceCreation", "CompilationUnit",
                            "ConditionalExpression", "ConstructorInvocation", "ContinueStatement", "DoStatement",
                            "EmptyStatement", "EnhancedForStatement", "EnumConstantDeclaration", "EnumDeclaration",
                            "ExpressionStatement", "FieldAccess", "FieldDeclaration", "ForStatement", "IfStatement",
                            "ImportDeclaration", "InfixExpression", "Initializer", "InstanceofExpression", "Javadoc",
                            "LabeledStatement", "LineComment", "MarkerAnnotation", "MemberRef", "MemberValuePair",
                            "MethodDeclaration", "MethodInvocation", "MethodRef", "MethodRefParameter", "Modifier",
                            "NormalAnnotation", "NullLiteral", "NumberLiteral", "PackageDeclaration",
                            "ParameterizedType", "ParenthesizedExpression", "PostfixExpression", "PrefixExpression",
                            "PrimitiveType", "QualifiedName", "QualifiedType", "ReturnStatement", "SimpleName",
                            "SimpleType", "SingleMemberAnnotation", "SingleVariableDeclaration", "StringLiteral",
                            "SuperConstructorInvocation", "SuperFieldAccess", "SuperMethodInvocation", "SwitchCase",
                            "SwitchStatement", "SynchronizedStatement", "TagElement", "TextElement", "ThisExpression",
                            "ThrowStatement", "TryStatement", "TypeDeclaration", "TypeDeclarationStatement",
                            "TypeLiteral", "TypeParameter", "VariableDeclarationExpression",
                            "VariableDeclarationFragment", "VariableDeclarationStatement", "WhileStatement",
                            "WildcardType"]
    syntax_tree_features_norm = ["NORM_" + feature for feature in syntax_tree_features]

    df_train = df_train.drop(syntax_tree_features, axis='columns')
    df_train = df_train.drop(syntax_tree_features_norm, axis='columns')
    df_train = df_train.drop(["SUM", "packagename", "plugin"], axis='columns')

    df_test = df_test.drop(syntax_tree_features, axis='columns')
    df_test = df_test.drop(syntax_tree_features_norm, axis='columns')
    df_test = df_test.drop(["SUM", "packagename", "plugin"], axis='columns')

    # Sets the training and testing sets for all analyses
    x_train = df_train.loc[:, df_train.columns != "post"].values
    y_train = df_train.loc[:, df_train.columns == "post"].values
    y_train = np.array([np.array(0) if value == 0 else np.array(1) for value in y_train])
    x_val = df_test.loc[:, df_test.columns != "post"].values
    y_val = df_test.loc[:, df_test.columns == "post"].values
    y_val = np.array([np.array(0) if value == 0 else np.array(1) for value in y_val])

    # set the adequate attribute list
    attr = df_train.drop('post', 1)
    global attribute_list
    attribute_list = list(attr.columns)

    # 1
    nmin = 15
    minleaf = 5
    nfeat = 41
    if load:
        filehandler = open("single_tree", 'rb')
        tree = pickle.load(filehandler)
    else:
        tree = tree_grow(x_train, y_train, nmin, minleaf, nfeat)
        filehandler = open("single_tree", 'wb')
        pickle.dump(tree, filehandler)
    y_pred_tree = tree_pred(x_val, tree)
    print(tree)
    print_metrics(f"Single Tree (nmin = {nmin}, minleaf = {minleaf}, nfeat= {nfeat})", y_val, y_pred_tree)

    #2
    m = 100
    if load:
        filehandler = open("bagging_trees", 'rb')
        trees = pickle.load(filehandler)
    else:
        trees = tree_grow_b(x_train, y_train, nmin, minleaf, nfeat, m)
        filehandler = open("bagging_trees", 'wb')
        pickle.dump(trees, filehandler)
    y_pred_bag = tree_pred_b(x_val, trees)
    print_metrics(f"Bagging (nmin = {nmin}, minleaf = {minleaf}, nfeat= {nfeat})", y_val, y_pred_bag)

    # 3
    m = 100
    nfeat = 6  # ~ square root of 41
    if load:
        filehandler = open("rf_trees", 'rb')
        trees = pickle.load(filehandler)
    else:
        trees = tree_grow_b(x_train, y_train, nmin, minleaf, nfeat, m)
        filehandler = open("rf_trees", 'wb')
        pickle.dump(trees, filehandler)
    y_pred_rf = tree_pred_b(x_val, trees)
    print_metrics(f"Random Forest (nmin = {nmin}, minleaf = {minleaf}, nfeat= {nfeat})", y_val, y_pred_rf)

    # Comparing simple tree with bagging
    mcnemar_test(y_val, y_pred_tree, y_pred_bag)

    # Comparing simple tree with random forest
    mcnemar_test(y_val, y_pred_tree, y_pred_rf)

    # Comparing bagging with random forest
    mcnemar_test(y_val, y_pred_bag, y_pred_rf)

def mcnemar_test(y_val, y_pred1, y_pred2):
    y_pred1_corr = np.equal(y_pred1, y_val)
    y_pred2_corr = np.equal(y_pred2, y_val)
    both_correct = np.count_nonzero(np.add(y_pred1_corr.astype(int), y_pred2_corr.astype(int)) == 2)
    both_false = np.count_nonzero(np.add(y_pred1_corr.astype(int), y_pred2_corr.astype(int)) == 0)
    corr_diff = np.subtract(y_pred1_corr.astype(int), y_pred2_corr.astype(int))
    y1_corr_y2_false = np.count_nonzero(corr_diff == 1)
    y1_false_y2_corr = np.count_nonzero(corr_diff == -1)

    mcnemar_table = [[both_correct, y1_corr_y2_false],[y1_false_y2_corr, both_false]]
    print(np.array(mcnemar_table))
    stat = mcnemar(np.array(mcnemar_table), exact=False, correction=True)
    print(str(stat) + "\n")

def analyse_credit_data():

    path = pathlib.Path(r"data/credit.txt")
    df = pd.read_csv(path)
    # create train/test sets
    x_train, x_val ,y_train , y_val = train_test_split(df[['age','married','house','income', 'gender']].values,df.label.values, test_size=0.10)

    print("\n", df)
    print("\ntest data:", x_val, y_val)
    print()

    # set the adequate attribute list
    global attribute_list
    attribute_list = list(['age','married','house','income', 'gender'])

    nmin = 1
    minleaf = 1
    nfeat = len(x_train[0, :])

    # grow a single tree and use it to predict the class label of the observation in the test set
    tree = tree_grow(x_train,y_train, nmin, minleaf, nfeat)
    y_pred = tree_pred(x_val, tree)
    print("\n\nSINGLE TREE\npredicted:\n", y_pred)
    print("real:\n", y_val)

    # use bagging to predict the class label of the observation in the test set
    m = 100
    tree = tree_grow_b(x_train,y_train, nmin, minleaf, nfeat, m)
    y_pred = tree_pred_b(x_val, tree)
    print("\n\nBAGGING\npredicted:\n", y_pred)
    print("real:\n", y_val)

    # grow a random forest and use it to predict the class label of the observation in the test set
    m = 100
    nfeat = 2       # ~ square root of 5
    tree = tree_grow_b(x_train,y_train, nmin, minleaf, nfeat, m)
    y_pred = tree_pred_b(x_val, tree)
    print("\n\nRANDOM FOREST\npredicted:\n", y_pred)
    print("real:\n", y_val)


def test_with_pima_data():
    path = pathlib.Path(r"data/pima.txt")
    df = pd.read_csv(path)
    # hint test
    nmin = 20
    minleaf = 5
    nfeat = len(list(df.columns)) - 1
    # grow a single tree and use it to predict the class label of the observation in the test set
    tree = tree_grow(df.iloc[:, 0:nfeat].values, df.iloc[:, nfeat].values, nmin, minleaf, nfeat)
    y_pred = tree_pred(df.iloc[:, 0:nfeat].values, tree)
    print(confusion_matrix(df.iloc[:, nfeat].values, y_pred))


def main():

    # t = time.time()
    print("Welcome! Pleas wait for the trees to be trained. This could take up to...")
    # analyse_credit_data()
    # test_with_pima_data()
    analyse_eclipse_data()
    # new_time = time.time()
    # print("tot time:", new_time-t)


if __name__ == '__main__':
    main()