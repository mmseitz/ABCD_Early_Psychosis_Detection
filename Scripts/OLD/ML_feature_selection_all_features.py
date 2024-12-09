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

###Output
# Index(['tfmri_mid_all_beh_hrwnfb_nt', 'tfmri_ma_alvcr_b_scs_tplh',
#        'tfmri_ma_acmvn_b_scs_aylh', 'tfmri_ma_acgml_b_scs_tplh',
#        'tfmri_sst_all_beh_incrgo_stdrt', 'tfmri_sst_all_beh_ssds_nt',
#        'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh',
#        'tfmri_sacsvcg_bscs_crbcxrh', 'tfmri_saigvcg_bscs_crbcxlh',
#        'tfmri_nback_all_28', 'tfmri_nback_all_164', 'tfmri_nback_all_178',
#        'tfmri_nback_all_215', 'tfmri_rec_all_beh_negface_pr',
#        'smri_thick_cdk_iftmlh', 'smri_thick_cdk_loccrh',
#        'smri_thick_cdk_paracnrh', 'smri_thick_cdk_smrh',
#        'smri_thick_cdk_meanlh', 'smri_vol_cdk_ptcatelh', 'smri_vol_cdk_loccrh',
#        'smri_vol_cdk_parstgrisrh', 'sit_scr_values_countnr_final',
#        'nihtbx_picture_agecorrected', 'nihtbx_picture_fc',
#        'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum',
#        'peq_ss_relational_aggs', 'peq_ss_overt_victim', 'weeks_preterm',
#        'income_3', 'speech_1', 'speech_4', 'speech_5'],
#       dtype='object')
# 0.8633681001687955
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
# Index(['tfmri_mid_all_beh_ntnfb_rate', 'tfmri_ma_acdn_b_scs_tplh',
#        'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_sst_all_beh_nrgo_nt',
#        'tfmri_sacgvf_bscs_tprh', 'tfmri_sacgvf_bscs_cderh',
#        'tfmri_saisvcg_bscs_crbwmlh', 'tfmri_saisvcg_bscs_tplh',
#        'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_nback_all_141',
#        'tfmri_nback_all_193', 'tfmri_nback_all_202', 'tfmri_nback_all_245',
#        'tfmri_rec_all_beh_place_dp', 'smri_thick_cdk_banksstslh',
#        'smri_thick_cdk_locclh', 'smri_thick_cdk_lingualrh',
#        'smri_thick_cdk_precnrh', 'smri_vol_cdk_frpolelh',
#        'smri_vol_cdk_rrmdfrrh', 'sit_scr_expr_mrt_finalrat',
#        'sit_scr_expr_mratdiff1', 'sit_scr_values_count3', 'ple_y_ss_total_bad',
#        'peq_ss_overt_victim'],
#       dtype='object')
# 0.8268776610485475
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
# Index(['tfmri_ma_alvcr_b_scs_tplh', 'tfmri_ma_acmvn_b_scs_aylh',
#        'tfmri_saasvcg_bscs_crbcxrh', 'tfmri_saigvis_bscs_crbcxlh',
#        'smri_thick_cdk_smrh', 'nihtbx_flanker_agecorrected',
#        'nihtbx_flanker_fc', 'ple_y_ss_affected_bad_sum',
#        'peq_ss_relational_aggs', 'peq_ss_overt_victim'],
#       dtype='object')
# 0.8128629733202328

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
# Index(['tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_hlpfb_nt',
#        'tfmri_ma_acmvn_b_scs_tprh', 'tfmri_sacsvcg_bscs_crbwmrh',
#        'tfmri_nback_all_155', 'smri_thick_cdk_banksstslh',
#        'smri_thick_cdk_paracnlh', 'sit_scr_expr_minitialrat',
#        'ple_y_ss_affected_bad_sum', 'peq_ss_overt_victim'],
#       dtype='object')
# 0.7904148086564383
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
['nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc', 'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum',
  'peq_ss_overt_victim', 'income_3', 'motor_5', 'speech_1', 'speech_5']
# %%
ok_cols = ['tfmri_mid_all_beh_hrwnfb_nt', 'tfmri_ma_alvcr_b_scs_tplh',
       'tfmri_ma_acmvn_b_scs_aylh', 'tfmri_ma_acgml_b_scs_tplh',
       'tfmri_sst_all_beh_incrgo_stdrt', 'tfmri_sst_all_beh_ssds_nt',
       'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh',
       'tfmri_sacsvcg_bscs_crbcxrh', 'tfmri_saigvcg_bscs_crbcxlh',
       'tfmri_nback_all_28', 'tfmri_nback_all_164', 'tfmri_nback_all_178',
       'tfmri_nback_all_215', 'tfmri_rec_all_beh_negface_pr',
       'smri_thick_cdk_iftmlh', 'smri_thick_cdk_loccrh',
       'smri_thick_cdk_paracnrh', 'smri_thick_cdk_smrh',
       'smri_thick_cdk_meanlh', 'smri_vol_cdk_ptcatelh', 'smri_vol_cdk_loccrh',
       'smri_vol_cdk_parstgrisrh', 'sit_scr_values_countnr_final',
       'nihtbx_picture_agecorrected', 'nihtbx_picture_fc',
       'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum',
       'peq_ss_relational_aggs', 'peq_ss_overt_victim', 'weeks_preterm',
       'income_3', 'speech_1', 'speech_4', 'speech_5', 'tfmri_mid_all_beh_ntnfb_rate', 'tfmri_ma_acdn_b_scs_tplh',
       'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_sst_all_beh_nrgo_nt',
       'tfmri_sacgvf_bscs_tprh', 'tfmri_sacgvf_bscs_cderh',
       'tfmri_saisvcg_bscs_crbwmlh', 'tfmri_saisvcg_bscs_tplh',
       'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_nback_all_141',
       'tfmri_nback_all_193', 'tfmri_nback_all_202', 'tfmri_nback_all_245',
       'tfmri_rec_all_beh_place_dp', 'smri_thick_cdk_banksstslh',
       'smri_thick_cdk_locclh', 'smri_thick_cdk_lingualrh',
       'smri_thick_cdk_precnrh', 'smri_vol_cdk_frpolelh',
       'smri_vol_cdk_rrmdfrrh', 'sit_scr_expr_mrt_finalrat',
       'sit_scr_expr_mratdiff1', 'sit_scr_values_count3', 'ple_y_ss_total_bad',
       'peq_ss_overt_victim', 'tfmri_ma_alvcr_b_scs_tplh', 'tfmri_ma_acmvn_b_scs_aylh',
       'tfmri_saasvcg_bscs_crbcxrh', 'tfmri_saigvis_bscs_crbcxlh',
       'smri_thick_cdk_smrh', 'nihtbx_flanker_agecorrected',
       'nihtbx_flanker_fc', 'ple_y_ss_affected_bad_sum',
       'peq_ss_relational_aggs', 'peq_ss_overt_victim', 'tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_hlpfb_nt',
       'tfmri_ma_acmvn_b_scs_tprh', 'tfmri_sacsvcg_bscs_crbwmrh',
       'tfmri_nback_all_155', 'smri_thick_cdk_banksstslh',
       'smri_thick_cdk_paracnlh', 'sit_scr_expr_minitialrat',
       'ple_y_ss_affected_bad_sum', 'peq_ss_overt_victim', 'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc', 'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum',
  'peq_ss_overt_victim', 'income_3', 'motor_5', 'speech_1', 'speech_5' ]
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
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_all_features_reduced.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_all_features_reduced.csv')
# %%
print(no_doubles)
# %%
['tfmri_mid_all_beh_hrwnfb_nt', 'tfmri_ma_alvcr_b_scs_tplh', 'tfmri_ma_acmvn_b_scs_aylh', 'tfmri_ma_acgml_b_scs_tplh',
  'tfmri_sst_all_beh_incrgo_stdrt', 'tfmri_sst_all_beh_ssds_nt', 'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh',
 'tfmri_sacsvcg_bscs_crbcxrh', 'tfmri_saigvcg_bscs_crbcxlh', 'tfmri_nback_all_28', 'tfmri_nback_all_164', 'tfmri_nback_all_178',
 'tfmri_nback_all_215', 'tfmri_rec_all_beh_negface_pr', 'smri_thick_cdk_iftmlh', 'smri_thick_cdk_loccrh', 
 'smri_thick_cdk_paracnrh', 'smri_thick_cdk_smrh', 'smri_thick_cdk_meanlh', 'smri_vol_cdk_ptcatelh', 'smri_vol_cdk_loccrh',
   'smri_vol_cdk_parstgrisrh', 'sit_scr_values_countnr_final', 'nihtbx_picture_agecorrected', 'nihtbx_picture_fc',
 'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs', 'peq_ss_overt_victim', 'weeks_preterm', 'income_3',
 'speech_1', 'speech_4', 'speech_5', 'tfmri_mid_all_beh_ntnfb_rate', 'tfmri_ma_acdn_b_scs_tplh', 'tfmri_ma_lvnfb_b_cds_clatcgelh',
 'tfmri_sst_all_beh_nrgo_nt', 'tfmri_sacgvf_bscs_tprh', 'tfmri_sacgvf_bscs_cderh', 'tfmri_saisvcg_bscs_crbwmlh',
   'tfmri_saisvcg_bscs_tplh', 'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_nback_all_141', 'tfmri_nback_all_193', 'tfmri_nback_all_202',
 'tfmri_nback_all_245', 'tfmri_rec_all_beh_place_dp', 'smri_thick_cdk_banksstslh', 'smri_thick_cdk_locclh', 'smri_thick_cdk_lingualrh', 
 'smri_thick_cdk_precnrh', 'smri_vol_cdk_frpolelh', 'smri_vol_cdk_rrmdfrrh', 'sit_scr_expr_mrt_finalrat', 'sit_scr_expr_mratdiff1', 
'sit_scr_values_count3', 'tfmri_saasvcg_bscs_crbcxrh', 'tfmri_saigvis_bscs_crbcxlh', 'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc', 
'tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_hlpfb_nt', 'tfmri_ma_acmvn_b_scs_tprh', 'tfmri_sacsvcg_bscs_crbwmrh', 'tfmri_nback_all_155',
 'smri_thick_cdk_paracnlh', 'sit_scr_expr_minitialrat', 'motor_5']
# %%
len(ok_cols)
# %%
len(no_doubles)
# %%
