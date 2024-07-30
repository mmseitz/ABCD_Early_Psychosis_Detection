#ML Feature Selection Script
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
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Model/X_train_matched_data_cog_only.csv')
    y_train = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Model/y_train_matched_data_cog_only.npy", allow_pickle= True)
    #y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Model/X_test_matched_data_cog_only.csv')
    y_test = np.load("//Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Model/y_test_matched_data_cog_only.npy", allow_pickle= True)
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
# Index(['tfmri_mid_all_beh_slpfb_nt', 'tfmri_ma_alrvn_b_scs_ayrh',
#        'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh',
#        'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_mrt',
#        'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_rt',
#        'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_ssds_nt',
#        'tfmri_sacgvf_bscs_crbwmrh', 'tfmri_sacgvf_bscs_crbcxrh',
#        'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbcxrh',
#        'tfmri_saisvcg_bscs_crbcxlh', 'tfmri_sacsvis_bscs_aalh',
#        'tfmri_saigvis_bscs_pdlh', 'tfmri_nback_all_22', 'tfmri_nback_all_53',
#        'tfmri_nback_all_96', 'tfmri_nback_all_124', 'tfmri_nback_all_178',
#        'tfmri_nback_all_202', 'tfmri_nback_all_214', 'tfmri_nback_all_223',
#        'tfmri_nback_all_245', 'tfmri_rec_all_beh_negface_pr',
#        'gdt_scr_values_single', 'sit_scr_expr_mrt_initialrat',
#        'sit_scr_expr_mratdiff1', 'sit_scr_expr_mratdiff2',
#        'sit_scr_values_count3', 'nihtbx_picture_agecorrected',
#        'lmt_scr_num_timed_out', 'lmt_scr_rt_correct'],
#       dtype='object')
# 0.7022435672828807
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
# Index(['tfmri_mid_all_beh_ntpfb_nt', 'tfmri_ma_acgml_b_scs_aylh',
#        'tfmri_ma_rpvnfb_b_cds_romdfrrh', 'tfmri_ma_lvnfb_b_cds_clatcgelh',
#        'tfmri_sst_all_beh_total_nt', 'tfmri_sst_all_beh_incrlg_rt',
#        'tfmri_sst_all_beh_crs_rt', 'tfmri_sacgvf_bscs_purh',
#        'tfmri_sacsvis_bscs_aarh', 'tfmri_saigvcg_bscs_pulh',
#        'tfmri_saigvcg_bscs_pdlh', 'tfmri_nback_all_134', 'tfmri_nback_all_164',
#        'tfmri_nback_all_185', 'tfmri_nback_all_193', 'tfmri_nback_all_215',
#        'tfmri_nback_all_246', 'tfmri_rec_all_beh_newnf_hr',
#        'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_negface_pr',
#        'gdt_scr_values_account_balance', 'sit_scr_values_count3'],
#       dtype='object')
# 0.6719790016927588
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
# Index(['tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_mrt',
#        'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_nt',
#        'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_sacgvf_bscs_pdrh',
#        'tfmri_nback_all_215', 'sit_scr_values_count3',
#        'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc'],
#       dtype='object')
# 0.6606629011737187

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
# Index(['tfmri_ma_alrvn_b_scs_ayrh', 'tfmri_ma_aclvn_b_scs_aylh',
#        'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_incrlg_rt',
#        'tfmri_sacsvcg_bscs_crbcxlh', 'tfmri_saisvcg_bscs_pdlh',
#        'tfmri_nback_all_51', 'tfmri_nback_all_134',
#        'tfmri_rec_all_beh_newpl_fa', 'sit_scr_values_count3'],
#       dtype='object')
# 0.6393771208150157
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
# ['tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_incrgo_rt', 
# 'tfmri_sst_all_beh_nrgo_rt', 'tfmri_nback_all_65', 'tfmri_nback_all_95', 'tfmri_nback_all_155', 'tfmri_nback_all_208',
#  'tfmri_nback_all_215', 'tfmri_rec_all_beh_newpl_hr', 'tfmri_rec_all_beh_newpl_fa', 'nihtbx_flanker_agecorrected',
#  'nihtbx_pattern_agecorrected', 'nihtbx_flanker_fc', 'nihtbx_pattern_fc']
# %%
ok_cols = ['tfmri_mid_all_beh_slpfb_nt', 'tfmri_ma_alrvn_b_scs_ayrh',
       'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh',
       'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_mrt',
       'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_rt',
       'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_ssds_nt',
       'tfmri_sacgvf_bscs_crbwmrh', 'tfmri_sacgvf_bscs_crbcxrh',
       'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbcxrh',
       'tfmri_saisvcg_bscs_crbcxlh', 'tfmri_sacsvis_bscs_aalh',
       'tfmri_saigvis_bscs_pdlh', 'tfmri_nback_all_22', 'tfmri_nback_all_53',
       'tfmri_nback_all_96', 'tfmri_nback_all_124', 'tfmri_nback_all_178',
       'tfmri_nback_all_202', 'tfmri_nback_all_214', 'tfmri_nback_all_223',
       'tfmri_nback_all_245', 'tfmri_rec_all_beh_negface_pr',
       'gdt_scr_values_single', 'sit_scr_expr_mrt_initialrat',
       'sit_scr_expr_mratdiff1', 'sit_scr_expr_mratdiff2',
       'sit_scr_values_count3', 'nihtbx_picture_agecorrected',
       'lmt_scr_num_timed_out', 'lmt_scr_rt_correct', 'tfmri_mid_all_beh_ntpfb_nt', 'tfmri_ma_acgml_b_scs_aylh',
       'tfmri_ma_rpvnfb_b_cds_romdfrrh', 'tfmri_ma_lvnfb_b_cds_clatcgelh',
       'tfmri_sst_all_beh_total_nt', 'tfmri_sst_all_beh_incrlg_rt',
       'tfmri_sst_all_beh_crs_rt', 'tfmri_sacgvf_bscs_purh',
       'tfmri_sacsvis_bscs_aarh', 'tfmri_saigvcg_bscs_pulh',
       'tfmri_saigvcg_bscs_pdlh', 'tfmri_nback_all_134', 'tfmri_nback_all_164',
       'tfmri_nback_all_185', 'tfmri_nback_all_193', 'tfmri_nback_all_215',
       'tfmri_nback_all_246', 'tfmri_rec_all_beh_newnf_hr',
       'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_negface_pr',
       'gdt_scr_values_account_balance', 'sit_scr_values_count3', 'tfmri_sst_beh_performflag',
       'tfmri_sst_all_beh_crgo_mrt',
       'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_nt',
       'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_sacgvf_bscs_pdrh',
       'tfmri_nback_all_215', 'sit_scr_values_count3',
       'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc', 'tfmri_ma_alrvn_b_scs_ayrh', 'tfmri_ma_aclvn_b_scs_aylh',
       'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_incrlg_rt',
       'tfmri_sacsvcg_bscs_crbcxlh', 'tfmri_saisvcg_bscs_pdlh',
       'tfmri_nback_all_51', 'tfmri_nback_all_134',
       'tfmri_rec_all_beh_newpl_fa', 'sit_scr_values_count3', 'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_incrgo_rt', 
'tfmri_sst_all_beh_nrgo_rt', 'tfmri_nback_all_65', 'tfmri_nback_all_95', 'tfmri_nback_all_155', 'tfmri_nback_all_208',
 'tfmri_nback_all_215', 'tfmri_rec_all_beh_newpl_hr', 'tfmri_rec_all_beh_newpl_fa', 'nihtbx_flanker_agecorrected',
 'nihtbx_pattern_agecorrected', 'nihtbx_flanker_fc', 'nihtbx_pattern_fc']
no_doubles = []
for l in ok_cols:
    if l in no_doubles:
        pass
    else:
        no_doubles.append(l)

# %%
X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Model/X_train_matched_data_cog_only.csv')
X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Model/X_test_matched_data_cog_only.csv')
# %%
X_train_reduced = X_train[no_doubles]
X_test_reduced = X_test[no_doubles]
# %%
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_anat_only_reduced.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_anat_only_reduced.csv')
# %%
print(no_doubles)
# %%
['tfmri_mid_all_beh_slpfb_nt', 'tfmri_ma_alrvn_b_scs_ayrh', 'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh',
'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_rt',
'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_ssds_nt', 'tfmri_sacgvf_bscs_crbwmrh', 'tfmri_sacgvf_bscs_crbcxrh', 
'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbcxrh', 'tfmri_saisvcg_bscs_crbcxlh', 'tfmri_sacsvis_bscs_aalh', 
'tfmri_saigvis_bscs_pdlh', 'tfmri_nback_all_22', 'tfmri_nback_all_53', 'tfmri_nback_all_96', 'tfmri_nback_all_124',
'tfmri_nback_all_178', 'tfmri_nback_all_202', 'tfmri_nback_all_214', 'tfmri_nback_all_223', 'tfmri_nback_all_245', 
'tfmri_rec_all_beh_negface_pr', 'gdt_scr_values_single', 'sit_scr_expr_mrt_initialrat', 'sit_scr_expr_mratdiff1', 
'sit_scr_expr_mratdiff2', 'sit_scr_values_count3', 'nihtbx_picture_agecorrected', 'lmt_scr_num_timed_out', 
'lmt_scr_rt_correct', 'tfmri_mid_all_beh_ntpfb_nt', 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_rpvnfb_b_cds_romdfrrh',
'tfmri_sst_all_beh_total_nt', 'tfmri_sst_all_beh_crs_rt', 'tfmri_sacgvf_bscs_purh', 'tfmri_sacsvis_bscs_aarh', 
'tfmri_saigvcg_bscs_pulh', 'tfmri_saigvcg_bscs_pdlh', 'tfmri_nback_all_134', 'tfmri_nback_all_164', 'tfmri_nback_all_185',
'tfmri_nback_all_193', 'tfmri_nback_all_215', 'tfmri_nback_all_246', 'tfmri_rec_all_beh_newnf_hr', 'tfmri_rec_all_beh_newnf_fa',
'gdt_scr_values_account_balance', 'tfmri_sst_all_beh_crlg_nt', 'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc', 
'tfmri_ma_aclvn_b_scs_aylh', 'tfmri_sacsvcg_bscs_crbcxlh', 'tfmri_saisvcg_bscs_pdlh', 'tfmri_nback_all_51', 'tfmri_rec_all_beh_newpl_fa', 
'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_incrgo_rt', 'tfmri_sst_all_beh_nrgo_rt', 'tfmri_nback_all_65', 'tfmri_nback_all_95',
'tfmri_nback_all_155', 'tfmri_nback_all_208', 'tfmri_rec_all_beh_newpl_hr', 'nihtbx_pattern_agecorrected', 'nihtbx_pattern_fc']
# %%
len(ok_cols)
# %%
