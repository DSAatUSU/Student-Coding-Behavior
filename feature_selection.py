# A random forest classifier will be fitted to compute the feature importances.
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
import pickle

with open("./data/feature_names.pkl", "rb") as fp:   # Unpickling
    feature_names = pickle.load(fp)

with open("./data/dataset.pkl", "rb") as fp:   # Unpickling
    temp_dataset_feat = pickle.load(fp)
X, y, _, _ = temp_dataset_feat

if __name__ == "__main__":
    feature_indexes = [f"{i}" for i in range(X.shape[1])]
    forest = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 5, min_samples_split= 5, n_estimators= 3, random_state= 1997)
    forest.fit(X, y)

    perm_important_feature_indexes = []
    important_feature_indexes = []
    num_times_to_run = 5
    for i in tqdm(range(num_times_to_run)):
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=feature_indexes) #feat 30 most important, go to the X dataframe and find 30th column
        important_feature_indexes.extend(forest_importances.sort_values(ascending=False)[forest_importances>0.005].index.values)


        #### Feature importance based on feature permutation
        # Feature importance based on feature permutation
        from sklearn.inspection import permutation_importance

        result = permutation_importance(forest, X, y, n_repeats=10, random_state=42, n_jobs=2)
        forest_importances = pd.Series(result.importances_mean+importances, index=feature_indexes).rename("Feature_Importance")
        perm_important_feature_indexes.extend(np.sort(forest_importances.sort_values(ascending=False)[forest_importances >0.02].index.values).tolist()) # low number means low importance 

    important_feature_indexes = np.array(np.unique(important_feature_indexes), dtype = int)
    perm_important_feature_indexes = np.array(np.unique(perm_important_feature_indexes))
    print(f"Length of Random Forest based Features Selection: {len(important_feature_indexes)}")
    print(f"Length of Permulation Based Features Selection: {len(perm_important_feature_indexes)}")
    important_feature_indexes = np.array(np.union1d(important_feature_indexes, perm_important_feature_indexes), dtype = int)
    print(f"Length of Unionized Set of Features: {len(important_feature_indexes)}")
    print("Feature Names: ", feature_names[important_feature_indexes])