"""
Utrecht University, Data Mining Assignment 2

Miguel Â. Simões Valente    6876005
Riccardo Bassani            6866840
Samuel Meyer                5648122
"""

# general
import platform
import os
import os.path
from pathlib import Path
import time
from collections import Counter
import itertools
import heapq
from collections import defaultdict

# np, pd
import numpy as np
import pandas as pd

# models
from sklearn.metrics import accuracy_score, precision_score, recall_score ,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif as mutinf
from sklearn.model_selection import GridSearchCV, cross_validate

# natural language
import spacy
from spacy.lang.en import English
from nltk import bigrams



spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
wordfreq = {}
# bi_wordfreq = {}              # local variable and normal method instead of global and .apply()

sparse_term_threshold = 0.0003           # threshold value relative frequency unigrams
bi_sparse_term_threshold = 0.00001        # threshold value relative frequency bigrams
# number_selected_features = 100        # better to move it inside specific model (as MNB)?
# feature_selection = False             # better to move it inside specific model (as MNB)?
FS_NB = True                            # whether to use feature selection for Multinomial Nayve Bayes



def reads_data(path, ):
    df = pd.DataFrame(columns=['text', 'fold', 'label'])
    counter = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            f = open(os.path.join(subdir, file), "r")
            if "truthful" in subdir:
                df.loc[counter] = [f.read(), subdir[-1], 0]

            else:
                df.loc[counter] = [f.read(), subdir[-1], 1]

            counter += 1

    return df


def preprocessing(text, lower=False):
    if lower:
        text = text.lower()
    tokens = tokenizer(text)
    token_list = []
    for token in tokens:
        token_list.append(token.text)
    return token_list


def word_frequency(text):
    for token in text:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1


def bigram_frequency(all_reviews):
    bigram_count = defaultdict(int)
    for review in all_reviews:
        for bigram in bigrams(review):
            bigram_count[bigram] +=1
    return bigram_count


def get_unigram_features(all_features, review):
    feature_array = np.zeros(len(all_features), dtype=np.int32)
    for i, u in enumerate(all_features):
        if u in review:
            feature_array[i] = review.count(u)
    return feature_array


def get_bigram_features(all_features, review):
    feature_array = np.zeros(len(all_features), dtype=np.int32)
    review_bigrams = Counter(bigrams(review))
    for i, b in enumerate(all_features):
        feature_array[i] = review_bigrams[b]
    return feature_array


"""
Best parameters on the test set are (max_features=12), but with cross-validation we get different values. 
Also, tuning on n_estimators is nonsense since the highest the number the better, it's just a matter of time
"""
def random_forest(X_folders, Y_folders, best_max_features, bigram=False):
    X_train_total = list(itertools.chain.from_iterable(X_folders[:4]))
    Y_train_total = list(itertools.chain.from_iterable(Y_folders[:4]))

    # the features with also bigrams are much more!
    if bigram:
        features_range = [best_max_features-2, best_max_features, best_max_features+2]           # if not good enough try higher numbers
    else:
        features_range = [best_max_features-2, best_max_features, best_max_features+2]
        # features_range = [20, 40, 50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 80, 100, 120]           # previous reuslts with range (40, 140, 20)
    parameters = {'max_features': features_range}                   # 'n_estimators': [1000] fixed

    optimized_forest = RandomForestClassifier(n_estimators=1000)
    clf_test = GridSearchCV(optimized_forest, parameters, cv=4)
    clf_test.fit(X_train_total, Y_train_total)
    # print(clf_test.cv_results_)
    rank_test = (clf_test.cv_results_['rank_test_score']).tolist()
    # print(rank_test)
    index = rank_test.index(min(rank_test))
    # print(index)
    best_parameters = clf_test.cv_results_["params"][index]
    # best_n_estimators = best_parameters["n_estimators"]
    best_max_features = best_parameters["max_features"]
    for i, p in enumerate(clf_test.cv_results_["params"]):
        print("n features = ", p["max_features"], "\tavg_accuracy =", clf_test.cv_results_["mean_test_score"][i], "\tstd_accuracy =", clf_test.cv_results_["std_test_score"][i] )

    clf = RandomForestClassifier(n_estimators=1000, max_features=best_max_features)
    X_test = X_folders[4]
    Y_test = Y_folders[4]
    clf = clf.fit(X_train_total, Y_train_total)
    predictions_test = clf.predict(X_test)
    predictions_training = clf.predict(X_train_total)   #+ str(best_n_estimators) + " trees, "
    print("Training Accuracy Random Forest ("  + str(
        best_max_features) + " features): " + str(accuracy_score(Y_train_total, predictions_training)))
    print("Test Accuracy Random Forest (" + str(
        best_max_features) + " features): " + str(accuracy_score(Y_test, predictions_test)))


"""
Looks like it makes sense, little improvment with cv, but it's something
"""
def decision_tree(X_folders, Y_folders):
    """
    Decision tree built through cost-complexity pruning
    """
    # build a tree for the ccp
    ccp_tree = DecisionTreeClassifier()
    # get alpha-beta values using ccp_tree
    X_train_total = list(itertools.chain.from_iterable(X_folders[:4]))
    Y_train_total = list(itertools.chain.from_iterable(Y_folders[:4]))
    path = ccp_tree.cost_complexity_pruning_path(X_train_total, Y_train_total)
    alphas = path["ccp_alphas"]
    betas = list()
    for i, a in enumerate(alphas[:-1]):
        betas.append(np.math.sqrt(a * alphas[i + 1]))
    betas.append(np.inf)

    # TODO test_fold_indices?
    # perform cross-validation to choose the best alpha value
    tree = DecisionTreeClassifier()
    parameters = {'ccp_alpha': betas}  # [x / 100.0 for x in range(1, 10, 1)]
    clf_test = GridSearchCV(tree, parameters, cv=4)
    clf_test.fit(X_train_total, Y_train_total)
    rank_test = (clf_test.cv_results_['rank_test_score']).tolist()
    index_best_alpha = rank_test.index(min(rank_test))
    bestGSalpha = betas[index_best_alpha]
    # print("Index of best GS alpha: ", index_best_alpha)
    print("Best GS alpha: ", bestGSalpha)

    clf = DecisionTreeClassifier(ccp_alpha=bestGSalpha)
    X_train = list(itertools.chain.from_iterable(X_folders[:4]))
    Y_train = list(itertools.chain.from_iterable(Y_folders[:4]))
    X_test = X_folders[4]
    Y_test = Y_folders[4]
    clf = clf.fit(X_train, Y_train)
    predictions_test = clf.predict(X_test)
    predictions_training = clf.predict(X_train)
    print("Training Accuracy Decision Tree with GS CCP:", accuracy_score(Y_train, predictions_training))
    print("Test Accuracy Decision Tree with GS CCP:", accuracy_score(Y_test, predictions_test))

    clf = DecisionTreeClassifier()
    X_train = list(itertools.chain.from_iterable(X_folders[:4]))
    Y_train = list(itertools.chain.from_iterable(Y_folders[:4]))
    X_test = X_folders[4]
    Y_test = Y_folders[4]
    clf = clf.fit(X_train, Y_train)
    predictions_test = clf.predict(X_test)
    predictions_training = clf.predict(X_train)
    print("Training Accuracy Decision Tree without GS CCP:", accuracy_score(Y_train, predictions_training))
    print("Test Accuracy Decision Tree without GS CCP:", accuracy_score(Y_test, predictions_test))


"""
Feature selection for unigrams is very bad. Accuracy drops drastically.
Is this because we did something wrong? Or maybe with bigrams it will be useful?
The very werid thing is that when feature selecting 8442 features (i.e. all the features), 
the accuracy goes from 0.8875 to 0.975!
"""
def naive_bayes(X_folders, Y_folders, bigram=False): # unigrams: threshold of 0.01 & 157 features gives test acc 0.91875. Bigrams make it worse
    print("Multinomial Naive Bayes\n")
    X_train_total = list(itertools.chain.from_iterable(X_folders[:4]))
    Y_train_total = list(itertools.chain.from_iterable(Y_folders[:4]))
    X_test = X_folders[4]
    Y_test = Y_folders[4]

    # don't really reduce the number of features if feature selection is not applied
    X_train_reduced = X_train_total
    X_test_reduced = X_test

    # if feature selection is applied, choose fn with cv
    if FS_NB or bigram:
        v = 4
        if bigram:
            # param = [70, 60, 50, 40, 30, 20, 10]
            param = range(100, len(X_test), 100)
        else:
            # param = [50, 40, 30, 20, 10]
            # param = range(155, 165, 1)
            param = [1489, 1000]
        errors = np.empty([len(param), v], dtype=float)
        for test_folder in range(v):
            X_train = list()
            Y_train = list()
            X_test = None
            Y_test = None
            for f in range(v):
                if f == test_folder:
                    X_test = X_folders[f]
                    Y_test = Y_folders[f]
                else:
                    # X_train = np.concatenate((X_train, X_folders[f]))
                    # Y_train = np.concatenate((Y_train, Y_folders[f]))
                    X_train = list(itertools.chain.from_iterable([X_train, X_folders[f]]))
                    Y_train = list(itertools.chain.from_iterable([Y_train, Y_folders[f]]))
            for i, n_features in enumerate(param):
                train_size = len(X_train)
                # X_all = np.concatenate((X_train, X_test))
                # Y_all = np.concatenate((Y_train, Y_test))
                X_all = list(itertools.chain.from_iterable([X_train, X_test]))
                Y_all = list(itertools.chain.from_iterable([Y_train, Y_test]))
                X_all_reduced = perform_feature_selection(X_all, Y_all, n_features)
                X_train_reduced = X_all_reduced[:train_size]
                X_test_reduced = X_all_reduced[train_size:]
                nb_network = MultinomialNB()
                nb_network.fit(X_train_reduced, Y_train)
                predictions = nb_network.predict(X_test_reduced)
                error_rate = 1 - accuracy_score(Y_test, predictions)
                # print("For fold", test_folder, "and", n_features, "features, error =", error_rate)
                errors[i][test_folder] = error_rate
        sum_errors = np.sum(errors, 1)
        for i, n in enumerate(param):
            avg_error = sum_errors[i] / v
            avg_acc = 1 - avg_error
            print("n =", n, "\tavg_error =", avg_error, "\tavg_accuracy =", avg_acc)
        best_n = param[np.argmin(sum_errors)]
        print("best_n:", best_n)
        train_size = len(X_train_total)
        X_all = list(itertools.chain.from_iterable([X_train_total, X_test]))
        Y_all = list(itertools.chain.from_iterable([Y_train_total, Y_test]))
        X_all_reduced = perform_feature_selection(X_all, Y_all, best_n)
        X_train_reduced = X_all_reduced[:train_size]
        X_test_reduced = X_all_reduced[train_size:]
    nb_network = MultinomialNB()
    nb_network.fit(X_train_reduced, Y_train_total)
    predictions_test = nb_network.predict(X_test_reduced)
    predictions_training = nb_network.predict(X_train_reduced)
    print("Training Accuracy MNB:", accuracy_score(Y_train_total, predictions_training))
    print("Test Accuracy MNB:", accuracy_score(Y_test, predictions_test))

def logistic_regression(X_folders, Y_folders, bigram = False):
    X_train = list(itertools.chain.from_iterable(X_folders[:4]))
    Y_train = list(itertools.chain.from_iterable(Y_folders[:4]))
    X_test = X_folders[4]
    Y_test = Y_folders[4]

    lambdas = np.arange(1, 200, 10)
    parameters = {'C': lambdas}
    
    log_reg = LogisticRegression(random_state=0)

    grid_log_reg = GridSearchCV(log_reg, parameters, cv=4)
    grid_log_reg.fit(X_train, Y_train)

    rank_test = (grid_log_reg.cv_results_['rank_test_score']).tolist()
    
    predictions = grid_log_reg.predict(X_test)

    if bigram:
        print(bigram)
    else:
        print("Unigram")
    print(f"The best C value is: {lambdas[np.argmax(rank_test)]}")
    print(f"Logistic Regression - Accuracy: {accuracy_score(predictions, Y_test)}")
    print(f"Logistic Regression - Precision: {precision_score(predictions, Y_test)}")
    print(f"Logistic Regression - Recall: {recall_score(predictions, Y_test)}")
    print(f"Logistic Regression - F1_score: {f1_score(predictions, Y_test)}")

    


def perform_feature_selection(X, Y, n):

    # Make list of mutual information scores of features and rank its indices using training folds only
    mutual_information_list = mutinf(X, Y, discrete_features=True)
    top_mutinf_indices = np.argpartition(mutual_information_list, -n)[-n:]
    X_reduced = np.array(X)[:, top_mutinf_indices]
    return X_reduced


def main():

    # load and pre-process data

    ##
    path = Path(r"op_spam_v1.4/negative_polarity")
    data_file = Path(r"data.csv")
    df = reads_data(path)
    df["text_processed"] = df["text"].apply(preprocessing)
    df.to_csv(data_file, index=False, header=True)
    df_bi = df.copy()
    df["text_processed"].apply(word_frequency)
    ##

    # most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)

    # less_freq = heapq.nsmallest(int(len(wordfreq)*sparse_term_threshold),wordfreq,key=wordfreq.get)
    for ech in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.005, 0.01, 0.05]:
        print(ech)
        sparse_term_threshold = ech
        print(len(wordfreq))
        most_freq = list(k for k, v in wordfreq.items() if v/len(wordfreq) > sparse_term_threshold)
        # build the vocabulary (unigram features)
        voc = list()
        for review in (df["text_processed"]):
            for word in review:
                if word not in voc and word in most_freq:
                    voc.append(word)
        print(len(voc))
        print(voc[0:10])

        # find all non-sparse bigrams
        bi_wordfreq = bigram_frequency(df_bi["text_processed"])
        df_bi["text_processed"].apply(bigram_frequency)
        non_sparse_bigrams = list(k for k, v in bi_wordfreq.items() if v / len(bi_wordfreq) > sparse_term_threshold)
        print(len(bi_wordfreq))
        print(len(non_sparse_bigrams))
        print(non_sparse_bigrams[0:10])

        # just check it makes sense
        # first_review = df["text_processed"][0]
        # print(first_review)
        # print(np.count_nonzero(get_unigram_features(voc, first_review)), "ones over", len(voc), "features")
        # print(np.count_nonzero(get_bigram_features(non_sparse_bigrams, first_review)), "ones over", len(non_sparse_bigrams), "features")

        # get X and Y for the training, for only unigrams and for also bigrams
        # (X=rows of features counts for each review. Y=label)
        X_unigrams = list()
        X_bigrams = list()
        for row in df["text_processed"]:
            unig = get_unigram_features(voc, row)
            big = get_bigram_features(non_sparse_bigrams, row)
            unig_big = np.concatenate((unig, big))
            X_unigrams.append(unig)
            X_bigrams.append(unig_big)
        Y = df["label"].to_list()

        # split data in folds to allow for cross-validation
        X_folders_unigrams = [[], [], [], [], []]
        X_folders_bigrams = [[], [], [], [], []]
        Y_folders = [[], [], [], [], []]
        for i, fold in enumerate(df["fold"]):
            X_folders_unigrams[int(fold) - 1].append(X_unigrams[i])
            X_folders_bigrams[int(fold) - 1].append(X_bigrams[i])
            Y_folders[int(fold) - 1].append(int(Y[i]))


        # logistic_regression(X_folders_unigrams, Y_folders)
        # logistic_regression(X_folders_bigrams, Y_folders, "Bigram")

        # if feature_selection:
        #     train_X = list(itertools.chain.from_iterable(X_folders[:4]))
        #     train_Y = list(itertools.chain.from_iterable(Y_folders[:4]))
        #
        #     # Make list of mutual information scores of features and rank its indices using training folds only
        #     mutual_information_list = mutinf(train_X, train_Y, discrete_features=True)
        #     # print(mutual_information_list)
        #     top_mutinf_indices = np.argpartition(mutual_information_list, -number_selected_features)[
        #                          -number_selected_features:]
        #     print(top_mutinf_indices)
        #     for topper in top_mutinf_indices:
        #         print(voc[topper])
        #         print(mutual_information_list[topper])
        #     X = np.array(X)[:, top_mutinf_indices]

        # # Unigrams
        start = time.time()
        # print("\n\nOnly unigrams as features:")
        # decision_tree(X_folders_unigrams, Y_folders)
        # after_dt = time.time()
        # print("Time for dt:", (after_dt-start)/60, "min\n")
        # random_forest(X_folders_unigrams, Y_folders)

        # n = int(np.sqrt(len(X_folders_unigrams[0][0])))
        # random_forest(X_folders_unigrams, Y_folders, n)

        # after_rf = time.time()
        # print("Time for rf:", (after_rf-after_dt)/60, "min\n")
        naive_bayes(X_folders_unigrams, Y_folders)
        after_nb = time.time()
        # # print("Time for nb:", (after_nb-after_rf)/60, "min")
        print("\nTot_time for unigrams:", (after_nb-start)/60, "min")
        #
        # # Unigrams and Bigrams
        start = time.time()
        # print("\n\nBoth unigrams and bigrams:\n")
        # decision_tree(X_folders_bigrams, Y_folders)
        # after_dt = time.time()
        # print("Time for dt:", (after_dt-start)/60, "min\n")

        # n = int(np.sqrt(len(X_folders_bigrams[0][0])))
        # random_forest(X_folders_unigrams, Y_folders, n, bigram=True)

        # random_forest(X_folders_bigrams, Y_folders, bigram=True)
        # after_rf = time.time()
        # print("Time for rf:", (after_rf-after_dt)/60, "min\n")
        # naive_bayes(X_folders_bigrams, Y_folders, bigram=True)
        # after_nb = time.time()
        # # print("Time for nb:", (after_nb-after_rf)/60)
        print("\nTot_time for bigrams:", (after_nb-start)/60, "min")


if __name__ == '__main__':
    main()
