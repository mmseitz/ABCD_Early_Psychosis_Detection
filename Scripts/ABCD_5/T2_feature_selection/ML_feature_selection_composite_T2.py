#ML Feature Selection Script (all features)
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
import scipy.stats as stats
from scipy.stats import randint
print('done')
# %%
####
# Import data....
def data():
    print("Loading data...", end="")
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/All_Data_Uniform_Model/X_train_all_data.csv')
    y_train = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/All_Data_Uniform_Model/y_train_all_data.npy", allow_pickle= True)
    #y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/All_Data_Uniform_Model/X_test_all_data.csv')
    y_test = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/All_Data_Uniform_Model/y_test_all_data.npy", allow_pickle= True)
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

# Index(['tfmri_ma_rpvnfb_b_cds_clcgelh', 'tfmri_ma_acgml_b_scs_tplh',
#        'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
#        'tfmri_sst_all_beh_ssds_nt', 'tfmri_sacgvf_bscs_crbwmlh',
#        'tfmri_sacgvf_bscs_tplh', 'tfmri_sacgvf_bscs_cdelh',
#        'tfmri_sacsvcg_bscs_crbwmlh', 'tfmri_sacsvcg_bscs_aarh',
#        'tfmri_saasvcg_bscs_pdlh', 'tfmri_saasvcg_bscs_crbcxrh',
#        'tfmri_saigvcg_bscs_crbwmlh', 'tfmri_nback_all_66',
#        'tfmri_nback_all_87', 'sit_scr_values_count3', 'lmt_scr_perc_correct',
#        'smri_vol_cdk_cdacatelh', 'smri_vol_cdk_cdmdfrrh',
#        'smri_vol_cdk_lobfrrh', 'smri_vol_cdk_periccrh',
#        'smri_vol_cdk_tmpolerh', 'smri_thick_cdk_pclh',
#        'smri_thick_cdk_insulalh', 'smri_thick_cdk_suplrh',
#        'ple_y_ss_total_number', 'ple_y_ss_total_good',
#        'peq_ss_relational_victim', 'peq_ss_reputation_victim',
#        'peq_ss_overt_victim', 'comc_ss_collective_capacity_p', 'disc_2_a',
#        'crime_1', 'crime_2', 'crime_5'],
#       dtype='object')
# 0.9106127974052503
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
# Index(['tfmri_mid_all_beh_slnfb_rate', 'tfmri_ma_lvnfb_b_cds_clatcgelh',
#        'tfmri_ma_acmvn_b_scs_ayrh', 'tfmri_sst_all_beh_crgo_nt',
#        'tfmri_sst_all_beh_incrs_mrt', 'tfmri_sacgvf_bscs_purh',
#        'tfmri_saisvcg_bscs_pdlh', 'tfmri_saisvcg_bscs_crbcxrh',
#        'tfmri_nback_all_87', 'smri_vol_cdk_periccrh',
#        'smri_thick_cdk_postcnrh', 'ple_y_ss_total_number',
#        'peq_ss_relational_victim', 'peq_ss_overt_victim', 'comc_ss_control_p',
#        'income_3', 'disc_3_a', 'disc_4_a', 'crime_2', 'crime_5'],
#       dtype='object')
# 0.88507170676982
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
# Index(['tfmri_sst_all_beh_crgo_stdrt', 'tfmri_nback_all_66',
#        'tfmri_nback_all_87', 'smri_vol_cdk_cuneusrh', 'ple_y_ss_total_number',
#        'ple_y_ss_total_good', 'peq_ss_relational_victim',
#        'peq_ss_reputation_victim', 'crime_2', 'crime_5'],
#       dtype='object')
# 0.8656379609209797

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
# Index(['tfmri_ma_lvnfb_b_cds_roagelh', 'tfmri_sacsvcg_bscs_crbwmrh',
#        'tfmri_nback_all_13', 'nihtbx_picture_agecorrected',
#        'ple_y_ss_total_number', 'peq_ss_relational_victim',
#        'peq_ss_reputation_victim', 'comc_ss_collective_capacity_p', 'disc_3_b',
#        'crime_5'],
#       dtype='object')
# 0.8314409734221054
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
['lmt_scr_avg_rt', 'lmt_scr_rt_correct', 'ple_y_ss_total_number', 'ple_y_ss_total_good', 'peq_ss_relational_victim',
 'peq_ss_overt_victim', 'income_3', 'speech_4', 'disc_2_a', 'disc_2_b', 'disc_3_b', 'crime_2', 'crime_5']
# %%
X_test = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_all_features/X_test_all_data_abcd5_T2.csv')
X_train = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_all_features/X_train_all_data_abcd5_T2.csv')
# %%
ok_cols = ['tfmri_ma_rpvnfb_b_cds_clcgelh', 'tfmri_ma_acgml_b_scs_tplh',
       'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
       'tfmri_sst_all_beh_ssds_nt', 'tfmri_sacgvf_bscs_crbwmlh',
       'tfmri_sacgvf_bscs_tplh', 'tfmri_sacgvf_bscs_cdelh',
       'tfmri_sacsvcg_bscs_crbwmlh', 'tfmri_sacsvcg_bscs_aarh',
       'tfmri_saasvcg_bscs_pdlh', 'tfmri_saasvcg_bscs_crbcxrh',
       'tfmri_saigvcg_bscs_crbwmlh', 'tfmri_nback_all_66',
       'tfmri_nback_all_87', 'sit_scr_values_count3', 'lmt_scr_perc_correct',
       'smri_vol_cdk_cdacatelh', 'smri_vol_cdk_cdmdfrrh',
       'smri_vol_cdk_lobfrrh', 'smri_vol_cdk_periccrh',
       'smri_vol_cdk_tmpolerh', 'smri_thick_cdk_pclh',
       'smri_thick_cdk_insulalh', 'smri_thick_cdk_suplrh',
       'ple_y_ss_total_number', 'ple_y_ss_total_good',
       'peq_ss_relational_victim', 'peq_ss_reputation_victim',
       'peq_ss_overt_victim', 'comc_ss_collective_capacity_p', 'disc_2_a',
       'crime_1', 'crime_2', 'crime_5', 'tfmri_mid_all_beh_slnfb_rate', 'tfmri_ma_lvnfb_b_cds_clatcgelh',
       'tfmri_ma_acmvn_b_scs_ayrh', 'tfmri_sst_all_beh_crgo_nt',
       'tfmri_sst_all_beh_incrs_mrt', 'tfmri_sacgvf_bscs_purh',
       'tfmri_saisvcg_bscs_pdlh', 'tfmri_saisvcg_bscs_crbcxrh',
       'tfmri_nback_all_87', 'smri_vol_cdk_periccrh',
       'smri_thick_cdk_postcnrh', 'ple_y_ss_total_number',
       'peq_ss_relational_victim', 'peq_ss_overt_victim', 'comc_ss_control_p',
       'income_3', 'disc_3_a', 'disc_4_a', 'crime_2', 'crime_5', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_nback_all_66',
       'tfmri_nback_all_87', 'smri_vol_cdk_cuneusrh', 'ple_y_ss_total_number',
       'ple_y_ss_total_good', 'peq_ss_relational_victim',
       'peq_ss_reputation_victim', 'crime_2', 'crime_5', 'tfmri_ma_lvnfb_b_cds_roagelh', 'tfmri_sacsvcg_bscs_crbwmrh',
       'tfmri_nback_all_13', 'nihtbx_picture_agecorrected',
       'ple_y_ss_total_number', 'peq_ss_relational_victim',
       'peq_ss_reputation_victim', 'comc_ss_collective_capacity_p', 'disc_3_b',
       'crime_5', 'lmt_scr_avg_rt', 'lmt_scr_rt_correct', 'ple_y_ss_total_number', 'ple_y_ss_total_good',
     'peq_ss_relational_victim', 'peq_ss_overt_victim', 'income_3', 'speech_4', 'disc_2_a', 'disc_2_b',
    'disc_3_b', 'crime_2', 'crime_5']
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
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_all_features_reduced_abcd5_T2.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_all_features_reduced_abcd5_T2.csv')
# %%
print(no_doubles)
# %%
['tfmri_ma_rpvnfb_b_cds_clcgelh', 'tfmri_ma_acgml_b_scs_tplh', 'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
 'tfmri_sst_all_beh_ssds_nt', 'tfmri_sacgvf_bscs_crbwmlh', 'tfmri_sacgvf_bscs_tplh', 'tfmri_sacgvf_bscs_cdelh',
 'tfmri_sacsvcg_bscs_crbwmlh', 'tfmri_sacsvcg_bscs_aarh', 'tfmri_saasvcg_bscs_pdlh', 'tfmri_saasvcg_bscs_crbcxrh',
 'tfmri_saigvcg_bscs_crbwmlh', 'tfmri_nback_all_66', 'tfmri_nback_all_87', 'sit_scr_values_count3', 'lmt_scr_perc_correct',
 'smri_vol_cdk_cdacatelh', 'smri_vol_cdk_cdmdfrrh', 'smri_vol_cdk_lobfrrh', 'smri_vol_cdk_periccrh', 'smri_vol_cdk_tmpolerh',
 'smri_thick_cdk_pclh', 'smri_thick_cdk_insulalh', 'smri_thick_cdk_suplrh', 'ple_y_ss_total_number', 'ple_y_ss_total_good',
 'peq_ss_relational_victim', 'peq_ss_reputation_victim', 'peq_ss_overt_victim', 'comc_ss_collective_capacity_p', 'disc_2_a',
 'crime_1', 'crime_2', 'crime_5', 'tfmri_mid_all_beh_slnfb_rate', 'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_ma_acmvn_b_scs_ayrh',
 'tfmri_sst_all_beh_incrs_mrt', 'tfmri_sacgvf_bscs_purh', 'tfmri_saisvcg_bscs_pdlh', 'tfmri_saisvcg_bscs_crbcxrh',
 'smri_thick_cdk_postcnrh', 'comc_ss_control_p', 'income_3', 'disc_3_a', 'disc_4_a', 'tfmri_sst_all_beh_crgo_stdrt',
 'smri_vol_cdk_cuneusrh', 'tfmri_ma_lvnfb_b_cds_roagelh', 'tfmri_sacsvcg_bscs_crbwmrh', 'tfmri_nback_all_13',
 'nihtbx_picture_agecorrected', 'disc_3_b', 'lmt_scr_avg_rt', 'lmt_scr_rt_correct', 'speech_4', 'disc_2_b']
# %%
len(ok_cols)
# %%
len(no_doubles)
# %%
