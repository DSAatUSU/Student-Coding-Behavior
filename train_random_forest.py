from IPython.display import clear_output
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import  classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.ensemble import RUSBoostClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RANSACRegressor
import pickle
import math
from collections import Counter
from sklearn.linear_model import LinearRegression
import numpy as np
import gc
from nlp_utlities import *
from collections import Counter
from imblearn.over_sampling import RandomOverSampler 

def minority_resampling(X, y):
    print('Original dataset shape %s' % Counter(np.array(y).reshape(-1)))
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res, y_res, 


gc.collect()

with open("./data/dataset.pkl", "rb") as fp:   # Unpickling
    temp_dataset_feat = pickle.load(fp)

X, y, y_reg, important_feature_indexes = temp_dataset_feat



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(X, y)
X_extract = []

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def correct_AUC(y_test, y_pred):
    x = np.linspace(0, 1, 100)
    total = len(y_test)
    correct = []
    false = []
    for delta in x:
        correct.append(np.sum((np.array(y_test) - np.array(y_pred)) > delta))
        false.append(total - correct[-1])
    return correct, false

def classifiers(selection, X_train, y_train):
    if selection == 1:
        clf = RandomForestClassifier(
        criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 5, min_samples_split= 5, n_estimators= 3, class_weight = {0:.5,1:.3,2:.2}
    )
    elif selection == 2:
        clf = DecisionTreeClassifier(
        criterion="gini", max_depth=8, random_state=1997
    )
    elif selection == 3:
        clf = SVC(
        kernel="rbf", C=1, gamma=0.1, random_state=1997
    )
    elif selection == 4:
        clf = LogisticRegression(
        solver="saga", C=1, random_state=1997, max_iter=10000
    )
    elif selection == 5:
        clf = RUSBoostClassifier(random_state=1997)
    elif selection == 6:
        kernel =1.0 * RBF(length_scale=1.0 , length_scale_bounds=(1e-1, 10.0))
        clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
    elif selection == 7:
        clf = RandomForestRegressor(
        criterion="squared_error", max_depth=8, n_estimators=500, max_features="log2", max_leaf_nodes=10, max_samples=4, min_samples_leaf=1, min_samples_split=2, oob_score=True, warm_start=True, bootstrap=True, random_state=1997
    )
    elif selection == 8:
        clf =  MLPRegressor(hidden_layer_sizes=(10), activation='tanh', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', max_iter=100, early_stopping=True)
    
    elif selection == 9:
        clf = RANSACRegressor(LinearRegression(),
            max_skips = 8,
            max_trials=100, 		# Number of Iterations
            min_samples=0.5, 		# Minimum size of the sample
            loss='absolute_error', 	# Metrics for loss
            residual_threshold=0.1,
            stop_n_inliers=250,
            stop_probability=0.1,
            stop_score=0.8# Threshold
		)

    clf.fit(X_train, y_train.ravel())  # fit to the training data
      # make predictions on the test data
    return clf
    # See how well the classifier does

def compute_metrics(selection, clf_array, X_train, y_train, X_test, y_test):
    y_pred_array = []    
    if selection in reg_list_of_classifiers:
        # Print labesl and predictions
        print(f"Number of Classifier in clf_array: {len(clf_array)}")
        for clf in clf_array:
            y_pred_array.append(clf.predict(X_test))

        y_pred = np.array(y_pred_array)
        #y_pred = [sigmoid(x) for x in y_pred]
        print(f"Labels, Predictions: {list(zip(np.round(y_test,2),np.round(y_pred,2)))}")
        #print(f"Predictions: {np.round(y_pred,2)}")

        # Plot y_test and y_pred
        plt.figure(figsize=(10, 6))
        plt.title(f"{clf.__class__.__name__} - y_test vs y_pred")
        plt.plot(np.round(y_test, 2),  marker='o', label="y_test")
        plt.plot(np.round(y_pred,2),  marker='o', label="y_pred")
        plt.legend()
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"Length of y_test: {len(y_test)} | Length of y_pred: {len(y_pred)}")
        correct, incorrect = correct_AUC(y_test, y_pred)
        # Plot correct against delta
        delta = np.linspace(0, 1, 100)
        plt.figure(figsize=(10, 6))
        plt.title(f"{clf.__class__.__name__} - Correct vs delta")
        plt.plot(delta, correct, label="Correct")
        plt.plot(delta, incorrect, label="Incorrect")
        plt.legend()

        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}") 

    else:
        for clf in clf_array:
            y_pred_array.append(clf.predict(X_test))
        tuples = list(zip(*y_pred_array))
        # Find max in each tuple
        y_pred = [max(Counter(t).items(), key = lambda ele : ele[1])[0] for t in tuples]
        
        y_test = np.ravel(y_test)
        y_pred = np.ravel(y_pred)
        test_conf_mat = confusion_matrix(y_test, y_pred)
        test_class_report = classification_report(y_test, y_pred, zero_division=1)
        test_macro_f1 = f1_score(y_test, y_pred, average='macro')

        y_pred = clf.predict(X_train)  # make predictions on the test data
        y_pred = np.ravel(y_pred)
        train_class_report = classification_report(y_train, y_pred, zero_division=1)
        train_conf_mat = confusion_matrix(y_train, y_pred)
        return test_macro_f1, train_class_report, test_class_report, train_conf_mat, test_conf_mat, clf_array
        
if __name__ == '__main__':
    list_of_classifiers = [1, 2]
    reg_list_of_classifiers = [7, 8, 9]
    macro_f1 = 0
    itteration = 0
    prev_macro_f1 = 0
    prev_clf_array = []
    prev_train_class_report = ""
    prev_test_class_report = ""
    prev_classifier_name = ""
    while itteration < 100:
        clear_output(wait=True)
        for i in list_of_classifiers:
            min_X = np.min(X)
            X_feats = X[:, important_feature_indexes]
            X_feats= np.log(X_feats - min_X + 1) # Change X_extract to X_sel and uncomment the keras MLP to extract features
            X_feats_reg = np.log(X_feats - min_X + 1)
            Y_feats = np.reshape(y, (len(y), 1))
            
            clf_array = []
            if i in reg_list_of_classifiers:
                indices = np.arange(len(X_feats_reg))
                X_train_main, X_test_main, y_train_main, y_test_main, indices_train, indices_test =  train_test_split(X_feats_reg, y_reg, indices, test_size=0.20, shuffle=True, random_state=42)
                clf_array.append(classifiers(i, X_train_main, y_train_main))
            else:
                indices = np.arange(len(X_feats))
                X_train_main, X_test_main, y_train_main, y_test_main, indices_train, indices_test =  train_test_split(X_feats, Y_feats, indices, stratify=Y_feats,test_size=0.20, random_state=42)
                X_train_main, y_train_main, assignment_order_res = minority_resampling(X_train_main, y_train_main)
                assignments_order = assignment_order_res
                clf_array.append(classifiers(i, X_train_main, y_train_main))
                
            macro_f1, train_class_report, test_class_report, train_conf_matrix, test_conf_matrix, clf_array = compute_metrics(i, clf_array, X_train_main, y_train_main, X_test_main, y_test_main)
            if prev_macro_f1 < macro_f1:
                prev_macro_f1 = macro_f1
                prev_train_class_report = train_class_report
                prev_test_class_report = test_class_report
                prev_classifier_name = clf_array[-1].__class__.__name__
                prev_clf_array = clf_array
            itteration += 1
    print(f"___________________________________________________________________Classifier Name: {prev_classifier_name} ___________________________________________________________________")

    print(f"Test Class Report: {prev_test_class_report}")
    #print(f"Test Confusion Matrix: {test_conf_matrix}")
    print(f"Train Class Report: {prev_train_class_report}")
    #print(f"Train Confusion Matrix: {train_conf_matrix}")
    print("________________________________________________________________________________________________________________________________________________________________________")
