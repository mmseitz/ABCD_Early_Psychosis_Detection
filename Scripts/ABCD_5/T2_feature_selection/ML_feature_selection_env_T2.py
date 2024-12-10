#ML Feature Selection Script (environmental data only)
# %%
####
print('Importing modules...', end='')
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
import os
import time
import joblib
from pickle import dump, load
from scipy.stats import randint
print('done')
# %%
####
# Import data....
def data():
    print("Loading data...", end="")
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_env/X_train_env_data_abcd5_T2.csv')
    y_train = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Env_Only_Uniform_Model/y_train_env_data.npy", allow_pickle= True)
    #y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_env/X_test_env_data_abcd5_T2.csv')
    y_test = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Env_Only_Uniform_Model/y_test_env_data.npy", allow_pickle= True)
    #y_test = y_test['group'].to_numpy()
    print("done!")
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = data()
#display(y_train)
# %%
####
#Import Modules
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# %%
####
##Sequential Forward Feature Selection
#Logistic Regression
sffs_LR = SequentialFeatureSelector(LogisticRegression(random_state=42),
                                    k_features = (5, 35), forward=True,
                                    floating=True, scoring='roc_auc',
                                    cv=3)
sffs_LR = sffs_LR.fit(X_train, y_train)

selected_features = X_train.columns[list(sffs_LR.k_feature_idx_)]
print(selected_features)
print(sffs_LR.k_score_)

###Output
# Index(['weeks_preterm', 'age_roll_over', 'age_walk', 'total_times_to_ER',
#        'ple_p_ss_affected_good_sum', 'ple_y_ss_total_number',
#        'ple_y_ss_total_good', 'ple_y_ss_affected_good_sum',
#        'ple_y_ss_affected_bad_sum', 'peq_ss_relational_victim',
#        'peq_ss_reputation_victim', 'peq_ss_overt_victim', 'dim_y_ss_mean',
#        'comc_ss_cohesion_p', 'comc_ss_control_p',
#        'comc_ss_collective_capacity_p', 'p_edu_2', 'p_edu_3', 'income_1',
#        'motor_1', 'motor_3', 'motor_5', 'speech_1', 'speech_4', 'speech_5',
#        'preterm_2', 'disc_2_a', 'disc_2_b', 'disc_3_a', 'disc_3_b', 'crime_1',
#        'crime_2', 'crime_3'],
#       dtype='object')
# 0.8266345037337182
# %%
####
#Random Forest Classifier
sffs_RF = SequentialFeatureSelector(RandomForestClassifier(),
                                    k_features = (5, 45), forward=True,
                                    floating=True, scoring='roc_auc',
                                    cv=3)
sffs_RF = sffs_RF.fit(X_train, y_train)

selected_features = X_train.columns[list(sffs_RF.k_feature_idx_)]
print(selected_features)
print(sffs_RF.k_score_)
###Output
# Index(['age_roll_over', 'age_first_word', 'ple_p_ss_total_number',
#        'ple_p_ss_total_good', 'ple_p_ss_total_bad', 'ple_y_ss_total_number',
#        'ple_y_ss_total_bad', 'ple_y_ss_affected_good_sum',
#        'peq_ss_relational_victim', 'peq_ss_reputation_victim',
#        'peq_ss_overt_aggression', 'peq_ss_overt_victim',
#        'peq_ss_relational_aggs', 'dim_y_ss_mean', 'p_edu_1', 'p_edu_2',
#        'p_edu_3', 'income_1', 'income_2', 'motor_5', 'speech_1', 'speech_2',
#        'speech_4', 'speech_5', 'preterm_1', 'preterm_2', 'disc_1_a',
#        'disc_1_b', 'disc_2_a', 'disc_2_b', 'disc_3_a', 'crime_1', 'crime_2',
#        'crime_4', 'crime_5'],
#       dtype='object')
# 0.8199155925445479
# %%
####
###Sequential Backwards Feature Selection
#Logistic Regression
sbfs_LR = SequentialFeatureSelector(LogisticRegression(random_state=42),
                                    k_features = 10, forward=False,
                                    floating=True, scoring='roc_auc',
                                    cv=3)
sbfs_LR = sbfs_LR.fit(X_train, y_train)

selected_features = X_train.columns[list(sbfs_LR.k_feature_idx_)]
print(selected_features)
print(sbfs_LR.k_score_)

###Output
# Index(['age_roll_over', 'ple_p_ss_affected_good_sum', 'ple_y_ss_total_number',
#        'ple_y_ss_total_good', 'ple_y_ss_affected_bad_sum',
#        'peq_ss_relational_victim', 'peq_ss_reputation_victim',
#        'peq_ss_overt_victim', 'disc_2_a', 'disc_3_b'],
#       dtype='object')
# 0.8146015828827121

# %%
#Random Forest Classifier
sbfs_RF = SequentialFeatureSelector(RandomForestClassifier(),
                                    k_features = 10, forward= False,
                                    floating=True, scoring='roc_auc',
                                    cv=3, verbose= 2)
sbfs_RF = sbfs_RF.fit(X_train, y_train)

selected_features = X_train.columns[list(sbfs_RF.k_feature_idx_)]
print(selected_features)
print(sbfs_RF.k_score_)

####Output
# Index(['ple_p_ss_total_number', 'ple_p_ss_affected_good_sum',
#        'ple_y_ss_total_number', 'ple_y_ss_affected_bad_sum',
#        'peq_ss_relational_victim', 'peq_ss_reputation_victim',
#        'peq_ss_overt_aggression', 'peq_ss_overt_victim', 'speech_2',
#        'speech_4'],
#       dtype='object')
# 0.7978012278920322
# %%
#####
#Recursive Feature Selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold
# %%
cv = StratifiedKFold(5)
visualizer = RFECV(LogisticRegression(), cv=cv, scoring='f1_weighted')
visualizer.fit(X_train, y_train)        # Fit the data to the visualizer
visualizer.show()
# %%
# Get list of chosen variables
cols = []
for i in range(len(visualizer.ranking_)):
    if visualizer.ranking_[i] == 1:
        cols.append(X_train.columns[i])

print(cols)
['weeks_preterm', 'age_roll_over', 'age_walk', 'total_times_to_ER', 'times_to_ER_past_year', 'ple_p_ss_total_number', 
 'ple_p_ss_total_good', 'ple_p_ss_total_bad', 'ple_p_ss_affected_good_sum', 'ple_p_ss_affected_bad_sum',
   'ple_y_ss_total_number', 'ple_y_ss_total_good', 'ple_y_ss_total_bad', 'ple_y_ss_affected_good_sum', 
'ple_y_ss_affected_bad_sum', 'peq_ss_relational_victim', 'peq_ss_reputation_aggs', 'peq_ss_reputation_victim',
 'peq_ss_overt_victim', 'comc_ss_cohesion_p', 'comc_ss_collective_capacity_p', 'p_edu_1', 'p_edu_3', 'income_1', 
 'income_2', 'income_3', 'motor_1', 'motor_2', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_4', 'speech_5',
 'preterm_1', 'preterm_2', 'disc_2_a', 'disc_2_b', 'disc_3_a', 'disc_3_b', 'disc_4_a', 'disc_4_b', 'crime_1', 'crime_2', 'crime_5']
# %%
ok_cols = ['weeks_preterm', 'age_roll_over', 'age_walk', 'total_times_to_ER',
       'ple_p_ss_affected_good_sum', 'ple_y_ss_total_number',
       'ple_y_ss_total_good', 'ple_y_ss_affected_good_sum',
       'ple_y_ss_affected_bad_sum', 'peq_ss_relational_victim',
       'peq_ss_reputation_victim', 'peq_ss_overt_victim', 'dim_y_ss_mean',
       'comc_ss_cohesion_p', 'comc_ss_control_p',
       'comc_ss_collective_capacity_p', 'p_edu_2', 'p_edu_3', 'income_1',
       'motor_1', 'motor_3', 'motor_5', 'speech_1', 'speech_4', 'speech_5',
       'preterm_2', 'disc_2_a', 'disc_2_b', 'disc_3_a', 'disc_3_b', 'crime_1',
       'crime_2', 'crime_3', 'age_roll_over', 'age_first_word', 'ple_p_ss_total_number',
       'ple_p_ss_total_good', 'ple_p_ss_total_bad', 'ple_y_ss_total_number',
       'ple_y_ss_total_bad', 'ple_y_ss_affected_good_sum',
       'peq_ss_relational_victim', 'peq_ss_reputation_victim',
       'peq_ss_overt_aggression', 'peq_ss_overt_victim',
       'peq_ss_relational_aggs', 'dim_y_ss_mean', 'p_edu_1', 'p_edu_2',
       'p_edu_3', 'income_1', 'income_2', 'motor_5', 'speech_1', 'speech_2',
       'speech_4', 'speech_5', 'preterm_1', 'preterm_2', 'disc_1_a',
       'disc_1_b', 'disc_2_a', 'disc_2_b', 'disc_3_a', 'crime_1', 'crime_2',
       'crime_4', 'crime_5', 'age_roll_over', 'ple_p_ss_affected_good_sum', 'ple_y_ss_total_number',
       'ple_y_ss_total_good', 'ple_y_ss_affected_bad_sum',
       'peq_ss_relational_victim', 'peq_ss_reputation_victim',
       'peq_ss_overt_victim', 'disc_2_a', 'disc_3_b', 'ple_p_ss_total_number', 'ple_p_ss_affected_good_sum',
       'ple_y_ss_total_number', 'ple_y_ss_affected_bad_sum',
       'peq_ss_relational_victim', 'peq_ss_reputation_victim',
       'peq_ss_overt_aggression', 'peq_ss_overt_victim', 'speech_2',
       'speech_4', 'weeks_preterm', 'age_roll_over', 'age_walk', 'total_times_to_ER', 
       'times_to_ER_past_year', 'ple_p_ss_total_number', 'ple_p_ss_total_good',
         'ple_p_ss_total_bad', 'ple_p_ss_affected_good_sum', 'ple_p_ss_affected_bad_sum', 
        'ple_y_ss_total_number', 'ple_y_ss_total_good', 'ple_y_ss_total_bad', 
    'ple_y_ss_affected_good_sum', 'ple_y_ss_affected_bad_sum', 'peq_ss_relational_victim',
     'peq_ss_reputation_aggs', 'peq_ss_reputation_victim', 'peq_ss_overt_victim', 
     'comc_ss_cohesion_p', 'comc_ss_collective_capacity_p', 'p_edu_1', 'p_edu_3', 
     'income_1', 'income_2', 'income_3', 'motor_1', 'motor_2', 'motor_4', 'motor_5', 
     'speech_1', 'speech_2', 'speech_4', 'speech_5', 'preterm_1', 'preterm_2', 'disc_2_a', 
     'disc_2_b', 'disc_3_a', 'disc_3_b', 'disc_4_a', 'disc_4_b', 'crime_1', 'crime_2', 'crime_5']
no_doubles = []
for l in ok_cols:
    if l in no_doubles:
        pass
    else:
        no_doubles.append(l)
# %%
X_train_reduced = X_train[no_doubles]
X_test_reduced = X_test[no_doubles]
# %%
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_env_reduced_abcd5_T2.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_env_reduced_abcd5_T2.csv')
# %%
print(no_doubles)
# %%
['weeks_preterm', 'age_roll_over', 'age_walk', 'total_times_to_ER', 'ple_p_ss_affected_good_sum', 
 'ple_y_ss_total_number', 'ple_y_ss_total_good', 'ple_y_ss_affected_good_sum',
 'ple_y_ss_affected_bad_sum', 'peq_ss_relational_victim', 'peq_ss_reputation_victim',
   'peq_ss_overt_victim', 'dim_y_ss_mean', 'comc_ss_cohesion_p', 'comc_ss_control_p',
 'comc_ss_collective_capacity_p', 'p_edu_2', 'p_edu_3', 'income_1', 'motor_1', 'motor_3',
 'motor_5', 'speech_1', 'speech_4', 'speech_5', 'preterm_2', 'disc_2_a', 'disc_2_b',
 'disc_3_a', 'disc_3_b', 'crime_1', 'crime_2', 'crime_3', 'age_first_word', 
'ple_p_ss_total_number', 'ple_p_ss_total_good', 'ple_p_ss_total_bad', 'ple_y_ss_total_bad',
 'peq_ss_overt_aggression', 'peq_ss_relational_aggs', 'p_edu_1', 'income_2', 'speech_2',
 'preterm_1', 'disc_1_a', 'disc_1_b', 'crime_4', 'crime_5', 'times_to_ER_past_year',
   'ple_p_ss_affected_bad_sum', 'peq_ss_reputation_aggs', 'income_3', 'motor_2', 
   'motor_4', 'disc_4_a', 'disc_4_b']
# %%
len(ok_cols)
# %%
len(no_doubles)
# %%
