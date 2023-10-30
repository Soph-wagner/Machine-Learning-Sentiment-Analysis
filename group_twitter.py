"""
Authors: Denis Onwualu and Sophia Wagner
Date: 10/29/2023
Description: This program is a classifier that predicts whether a movie review is positive or negative. 
"""

import numpy as np

from string import punctuation

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle


######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.

    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """

    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    np.savetxt(outfile, vec)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """

    word_list = {}
    with open(infile, 'r') as fid:
        # part 1-1: process each line to populate word_list
        i = 0
        for line in fid:
            words = extract_words(line)
            for word in words:
                if word not in word_list:
                    word_list[word] = i
                    i += 1
                
    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))

    with open(infile, 'r') as fid:
        ### ========== TODO: START ========== ###
        # part 1-2: process each line to populate feature_matrix
        
        #My code Below doesnt work :(
        # for line in fid:
        #     words = extract_words(line)
        #     row = 0
        #     for word in words:

        #         if word in word_list:
        #             i = list(word_list).index(word)

        #             feature_matrix[row,i] = 1
        #         else:
        #             print('wrong word my friend')
        #     row += 1

        #Sophia (My partner) code below works :)
        text = fid.read()
        lines = text.split('\n')
        for line in lines:
            words_per_line = extract_words(line)
            for word in words_per_line:
                if word in word_list:
                    feature_matrix[lines.index(line), word_list[word]] = 1
                else:
                    print("This word is not in the dictionary: " + word,
                           "located at line: ", lines.index(line))

            
                    
            
        ### ========== TODO: END ========== ###

    return feature_matrix


def test_extract_dictionary(dictionary):
    err = 'extract_dictionary implementation incorrect'

    assert len(dictionary) == 1811

    exp = [('2012', 0),
           ('carol', 10),
           ('ve', 20),
           ('scary', 30),
           ('vacation', 40),
           ('just', 50),
           ('excited', 60),
           ('no', 70),
           ('cinema', 80),
           ('frm', 90)]
    act = [sorted(dictionary.items(), key=lambda it: it[1])[i] for i in range(0, 100, 10)]
    assert exp == act


def test_extract_feature_vectors(X):
    err = 'extract_features_vectors implementation incorrect'

    assert X.shape == (630, 1811)
    
    exp = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    act = X[:10,:10]
    assert (exp == act).all()


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric='accuracy'):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1

    # part 2-1: compute classifier performance with sklearn metrics
    # hint: sensitivity == recall
    # hint: use confusion matrix for specificity (use the labels param)
    if metric == 'accuracy':
        return metrics.accuracy_score(y_true, y_label)
    elif metric == 'f1_score':
        return metrics.f1_score(y_true, y_label)
    elif metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_label)
    elif metric == 'precision':
        return metrics.precision_score(y_true, y_label)
    elif metric == 'sensitivity':
        return metrics.recall_score(y_true, y_label)
    elif metric == 'specificity':
        cm = metrics.confusion_matrix(y_true, y_label, labels = [1, -1]) #labels = [1, -1] is [positive, negative]
        #print(cm)
        #print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        
        tp = cm[0,0]
        fn = cm[0,1]
        fp = cm[1,0]
        tn = cm[1,1]  
        #return tp, fp, fn, tn  
        return tn / (tn + fp)    

    return 0


def test_performance():
    """Ensures performance scores are within epsilon of correct scores."""

    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    y_pred = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
               2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']
    scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]

    import sys
    eps = sys.float_info.epsilon
    print(eps)
    for i, metric in enumerate(metrics):
        assert abs(performance(y_true, y_pred, metric) - scores[i]) < eps, \
            (metric, performance(y_true, y_pred, metric), scores[i])


def cv_performance(clf, X, y, kf, metric='accuracy'):
    """
    Splits the data, X and y, into k folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """

    scores = []
    for train, test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make "continuous-valued" predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score):
            scores.append(score)
    return np.array(scores).mean()


def select_param_linear(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that maximizes the average k-fold CV performance.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure

    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """

    print('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)
    
    # part 2-3: select optimal hyperparameter using cross-validation
    # hint: create a new sklearn linear SVC for each value of C

    C_score = 'initial'
    Optimal_C = 0

    #For each value of C, run the cv_performance and create an array of scores.
    for C_value in C_range:
        LinearSVC = SVC(kernel = 'linear', C = C_value)
        score = cv_performance(LinearSVC, X , y , kf , metric)
        print('For a value of C = ' + str(C_value) + ', the score = ' + str(score))
        
        if C_score == 'initial' or score > C_score:
        
            C_score = score
            Optimal_C = C_value
   

    print('Optimal C = ' + str(Optimal_C))
    return Optimal_C
    ### ========== TODO: END ========== ###


def select_param_rbf(X, y, kf, metric='accuracy'):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure

    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """

    print('RBF SVM Hyperparameter Selection based on ' + str(metric) + ':')

 
    ### ========== TODO: START ========== ###
    # (Optional) part 3-1: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    Gamma_range = 10.0 ** np.arange(-3, 3)
    Grid_score = 'initial'

    for C_value in C_range:
            for Gamma_value in Gamma_range:
                RBFKernelSVC = SVC(kernel='rbf', C = C_value, gamma = Gamma_value)
                score = cv_performance(RBFKernelSVC, X , y , kf , metric)
                print('For a value of C, Gamma = ' + str(C_value) + ' , ' + str(Gamma_value) + ': the score = ' + str(score))
            
                if Grid_score == 'initial' or score > Grid_score:
                
                    Grid_score = score
                    Optimal_C = C_value
                    Optimal_Gamma = Gamma_value



    print('Optimal C , Gamma = ' + str(Optimal_C) + ' , ' + str(Optimal_Gamma))
    return Optimal_C, Optimal_Gamma
    ### ========== TODO: END ========== ###


def performance_CI(clf, X, y, metric='accuracy'):
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
        lower, upper -- tuple of floats, confidence interval
    """

    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric)

    # part 4-2: use bootstrapping to compute 95% confidence interval
    # hint: use np.random.randint(...) to get a random sample from y
    # hint: lower and upper are the values at 2.5% and 97.5% of the scores

    #Create an array of 1000 scores
    scores = []
    for i in range(1000):
        idx = np.random.randint(0, len(y), len(y))
        random_y = y_pred[idx]
        score = performance(y[idx], random_y, metric)
        scores.append(score)

    mean = np.mean(scores)
    std = np.std(scores)

    #upper_bound = int(mean + (1.96 * std))
    #lower_bound = int(mean - (1.96 * std))
    ## OR IS IT ##
    # upper_bound = int(mean + (2*std))
    # lower_bound = int(mean - (2*std))

    # DO I NEED TO SORT THE SCORES ARRAY? 
    scores.sort()

    #upper bound of 95% confidence interval
    upper_bound = int(0.975 * 1000)
    lower_bound = int(0.025 * 1000)

    lower_CI, higher_CI = scores[lower_bound], scores[upper_bound]

    return score, lower_CI, higher_CI


######################################################################
# main
######################################################################

def main():
    # read the tweets and its labels
    dictionary = extract_dictionary('../data/tweets.txt')
    test_extract_dictionary(dictionary)
    
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    test_extract_feature_vectors(X)

    y = read_vector_file('../data/labels.txt')

    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=0)

    # set random seed
    np.random.seed(1234)

    # split the data into training (training + cross-validation) and testing set
    X_train, X_test = X[:560], X[560:]
    y_train, y_test = y[:560], y[560:]

    metric_list = ['accuracy', 'f1_score', 'auroc', 'precision', 'sensitivity', 'specificity']

    ## ========== TODO: START ========== ###
    #test_performance()

    # part 2-2: create stratified folds (5-fold CV)

    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    # C = select_param_linear(X, y, SKF, metric='accuracy')
  
    # part 2-4: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
  
    #for metric in list, go through and pring C values, then decide which ones highest
    #Prints out every single value for C for the 'linear' SVC and then prints highest score
    for metric in metric_list:
        c = select_param_linear(X_train, y_train, SKF, metric = metric)
        print(c)

    # # (Optional) part 3-2: for each metric, select optimal hyperparameter for RBF-SVM using CV
    # #Prints out every single value for C and gamma for the 'rbf' SVC and then prints highest score
    # for metric in metric_list:
    #     select_param_rbf(X_train, y_train, SKF, metric = metric)

    # part 4-1: train linear-kernal SVM with selected hyperparameters
    LinearSVC = SVC(kernel = 'linear', C = c)
    LinearSVC.fit(X_train,y_train)
    # part 4-3: use bootstrapping to report performance on test data
    for metric in metric_list:
        print(metric)
        score, lower_CI, higher_CI = performance_CI(LinearSVC, X_test, y_test, metric)
        print('For a ' + str(metric) + ' score of ' + str(score) + ', the 95% confidence interval is ' + str(lower_CI) + ' to ' + str(higher_CI))


    # part 5: identify important features (hint: use best_clf.coef_[0])

    #Printing the indices of the 10 most important features for the first class (positive???)
    best_pos_feats = np.argsort(LinearSVC.coef_[0])[-10:]
    print(best_pos_feats)
    print('+++++++++++++++++++++++++++++++++++++++++')

    worst_feats = np.argsort(LinearSVC.coef_[0])[:10]
    print(worst_feats)
    print('+++++++++++++++++++++++++++++++++++++++++')

    # how do I print the actual features given the incides?
    for i in best_pos_feats:
        print(list(dictionary)[i])

    print('++++++++++ NOW PRINTING THE WORST WORDS +++++++++++')

    for i in worst_feats:
        print(list(dictionary)[i])


    #I want every line with a 1 in index 0 of the feature matrix

    print('attempting to print every tweet talking about 2012')
    num_of_2012 = 0
    pos_review = 0
    neg_review = 0
    for i in range(len(X)):
        if X[i][0] == 1:
            num_of_2012 += 1
            print(i)
            # print('label of corresponding tweet: ')
            # print(y[i])
            if y[i] == 1:
                pos_review += 1
            if y[i] == -1:
                neg_review += 1
    print('total positive reviews: ', pos_review)
    print('total negative reviews: ', neg_review)
    print('total tweets about 2012: ', num_of_2012)



    ### ========== TODO: END ========== ###

    ### ========== TODO: START ========== ###
    # part 6: (optional) contest!
    # uncomment out the following, and be sure to change the filename
    """
    X_held = extract_feature_vectors('../data/held_out_tweets.txt', dictionary)
    y_pred = best_clf.decision_function(X_held)
    write_label_answer(y_pred, '../data/YOUR_USERNAME_twitter.txt')
    """
    ### ========== TODO: END ========== ###


if __name__ == '__main__':
    main()
