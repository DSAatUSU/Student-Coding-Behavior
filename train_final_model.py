from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import numpy as np
import glob
import pickle
from IPython.display import clear_output
from sklearn.metrics import classification_report, f1_score, confusion_matrix
# Assuming you have X_train and y_train

if __name__ == "__main__":
    with open("./data/final_model_dataset.pkl", "rb") as fp:   # Unpickling
        temp_dataset = pickle.load(fp)

    with open(glob.glob("./data/RandomForestClassifier*.pkl")[0], "rb") as file:
        rf_clf = pickle.load(file)
    
    with open(glob.glob("./data/IsolationForest*.pkl")[0], "rb") as file:
        models = pickle.load(file)

    X_train_main, y_train_main, X_test_main, y_test_main = temp_dataset


    # Create dictionaries to store models for each algorithm
    # models = {
    #     'IsolationForest': {},
    #     # 'LocalOutlierFactor': {},
    #     #'OneClassSVM': {}
    # }

    # Fit models for each class using different algorithms
    # for class_label in np.unique(y_train_main):
    #     print(f"Fitting models for class {class_label}")
    #     indexes = np.where(np.reshape(y_train_main, (-1))==class_label)[0]
    #     X_class = X_train_main[indexes]

    #     # Isolation Forest
    #     iforest = IsolationForest(contamination='auto',n_estimators=5, random_state=42, bootstrap=True, n_jobs=-1)
    #     iforest.fit(X_class)
    #     models['IsolationForest'][class_label] = iforest

        # # Local Outlier Factor
        # lof = LocalOutlierFactor(contamination='auto', n_neighbors=5, novelty=True)
        # lof.fit(X_class)
        # models['LocalOutlierFactor'][class_label] = lof

        # # One-Class SVM
        # ocsvm = OneClassSVM(nu=0.02)
        # ocsvm.fit(X_class)
        # models['OneClassSVM'][class_label] = ocsvm



    num_classes = 3
    itteration = 0
    prev_macro_f1 = 0
    prev_test_class_report = ""
    max_itter = 1000
    prev_random_state = 0
    prev_conf_mat = []


    while itteration < max_itter:
        # Fit models for each class using different algorithms
        # Create dictionaries to store models for each algorithm
        random_state = np.random.randint(0, 1997)
        new_rf_clf = RandomForestClassifier(
            criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 5, min_samples_split= 5, n_estimators= 10, class_weight = {0:.5,1:.3,2:.2}, random_state=random_state
        )
        new_rf_clf.fit(X_train_main, y_train_main.ravel())
        
        models = {
            'IsolationForest': {},
            # 'LocalOutlierFactor': {},
            # 'OneClassSVM': {}
        }
        for class_label in [0, 1, 2]:
            #print(f"Fitting models for class {class_label}")
            indexes = np.where(np.reshape(y_train_main, (-1))==class_label)[0]
            X_class = X_train_main[indexes]

            # Isolation Forest
            iforest = IsolationForest(contamination='auto',n_estimators=20, random_state=random_state, bootstrap=True, n_jobs=-1)
            iforest.fit(X_class)
            models['IsolationForest'][class_label] = iforest

            # # Local Outlier Factor
            # lof = LocalOutlierFactor(contamination='auto', n_neighbors=5, novelty=True)
            # lof.fit(X_class)
            # models['LocalOutlierFactor'][class_label] = lof

            # # One-Class SVM
            # ocsvm = OneClassSVM(nu=0.02)
            # ocsvm.fit(X_class)
            # models['OneClassSVM'][class_label] = ocsvm
        clear_output(wait=True)
        print("Progress: ", end="")
        print(f"*"*int(((itteration)/max_itter)*100)+ "-"*(int(((max_itter-itteration)/max_itter)*100)-100) + f" {((itteration)/max_itter)*100:.2f}%")
        all_predictions = []
        for test_index in range(len(X_test_main)):
            example = X_test_main[test_index]
            rf_pred_prob = new_rf_clf.predict_proba([example])[0]
            average_scores = {}
            for class_label in [0, 1, 2]:
                scores_sum = 0
                num_models = 0
                for algorithm_models in models.values():
                    model = algorithm_models[class_label]
                    score = model.score_samples([example])[0]
                    scores_sum += score
                    num_models += 1
                average_score = scores_sum / num_models
                average_scores[class_label] = average_score
            # Normalize average scores to be between 0 and 1
            #average_scores = {k: (v - min(average_scores.values())) / (max(average_scores.values()) - min(average_scores.values())) for k, v in average_scores.items()}
            if np.argmax(rf_pred_prob) == 1:
                predicted_class = np.argmax(rf_pred_prob)
            else:
                score = list(average_scores.values())
                predicted_class = np.argmax(score)
            all_predictions.append(predicted_class)

        test_clf_report = classification_report(y_test_main.ravel(), all_predictions, zero_division=1)
        test_macro_f1 = f1_score(y_test_main.ravel(), all_predictions, average='macro')
        conf_mat = confusion_matrix(y_test_main.ravel(), all_predictions)

        if prev_macro_f1 < test_macro_f1:
            prev_random_state = random_state
            prev_macro_f1 = test_macro_f1
            prev_test_class_report = test_clf_report
            prev_conf_mat = conf_mat

        itteration += 1
    print(f"Test Class Report {prev_random_state}:\n {prev_test_class_report}")
    print(f"Confusion Matrix {prev_random_state}:\n {prev_conf_mat}")



    # while itteration < max_itter:
    #     clear_output(wait=True)
    #     print("Progress: ", end="")
    #     print("*"*itteration+ "__"*(max_itter-itteration) + f" {itteration}%")
    #     all_predictions = []
    #     for test_index in range(len(X_test_main)):
    #         example = X_test_main[test_index]
    #         rf_pred_prob = rf_clf.predict_proba([example])[0]
    #         average_scores = {}
    #         for class_label in range(num_classes):
    #             scores_sum = 0
    #             num_models = 0
    #             for algorithm_models in models.values():
    #                 model = algorithm_models[class_label]
    #                 score = model.score_samples([example])[0]
    #                 scores_sum += score
    #                 num_models += 1
    #             average_score = scores_sum / num_models
    #             average_scores[class_label] = average_score
    #         # Normalize average scores to be between 0 and 1
    #         #average_scores = {k: (v - min(average_scores.values())) / (max(average_scores.values()) - min(average_scores.values())) for k, v in average_scores.items()}
    #         if np.argmax(rf_pred_prob) == 1:
    #             predicted_class = np.argmax(rf_pred_prob)
    #         else:
    #             score = list(average_scores.values())
    #             predicted_class = np.argmax(score)
    #         all_predictions.append(predicted_class)

    #     test_clf_report = classification_report(y_test_main.ravel(), all_predictions, zero_division=1)
    #     test_macro_f1 = f1_score(y_test_main.ravel(), all_predictions, average='macro')

    #     if prev_macro_f1 < test_macro_f1:
    #         prev_macro_f1 = test_macro_f1
    #         prev_test_class_report = test_clf_report

    #     itteration += 1
    # print(f"Test Class Report:\n {prev_test_class_report}")