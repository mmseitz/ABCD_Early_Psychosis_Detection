#This script preprocesses environmental, cognitive (behavioral and fMRI), and anatomical data that will be fed into the Scikit-Learn ML models 
# %%
####
#Import libraries
print('Importing Modules...', end="")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import imblearn
#from imblearn.over_sampling import SMOTE
import scipy
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from scipy import stats
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns

print('done!')
# %%
#### Import raw data
print('Importing raw data...', end="")
abcd = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/master_csvs_uniform/uniform_abcd.csv')
abcd = abcd.drop(columns=['Unnamed: 0'])
#abcd shape (1357, 514) #HR group = 647
# %%
""" DEFINE FEATURE CATEGORIES """
# This script is set up as follows:
# 1. Split data in to train / test
# 2. Identify brain data (including average across hemispheres). At this point, if you want to include a continuous
# behavior predictor ('num_predictors'), identify it separately and include it in the Feature Data Pipeline
# 2. Regress confounds on to brain data.
num_confound_cols = ['interview_age']
cat_confound_cols = ['sex', 'race']

# Predictive Features
cat_predictors = ['p_edu', 'income', 'motor_dev_comparison', 'speech_dev_comparison']
oth_predictors = ['ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum', 'peq_ss_overt_aggression', 'peq_ss_relational_aggs', 'peq_ss_overt_victim', 'total_substances_used',
              'weeks_preterm', 'injury_totals']
cort_cols = list(abcd)[2:-15]

#dropnans
abcd = abcd.dropna()
# %%
""" DEFINE AND APPLY TRAIN/TEST SPLIT """
print('Define train/test split...', end="")
class TrainTestSplit(BaseEstimator, TransformerMixin):
    def __init__(self, label, test_size):
        self.test_size = test_size
        self.label = label
        self.X_train = pd.DataFrame(data=None)
        self.X_test = pd.DataFrame(data=None)
        self.y_train = pd.DataFrame(data=None)
        self.y_test = pd.DataFrame(data=None)
        self.train_ind = pd.DataFrame(data=None)
        self.test_ind = pd.DataFrame(data=None)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        label = X[self.label]
        features = X.drop([self.label], axis=1)
        X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(features, label, X.subjectkey, test_size=self.test_size, random_state=42)
        self.X_train = X_train.drop(['subjectkey'], axis=1)
        self.X_test = X_test.drop(['subjectkey'], axis=1)
        self.y_train = y_train
        self.y_test = y_test
        self.train_ind = train_ind
        self.test_ind = test_ind
        return self.X_train, self.X_test, self.y_train, self.y_test, self.train_ind, self.test_ind

train_test = TrainTestSplit('group', 0.2)
X_train, X_test, y_train, y_test, train_ind, test_ind = train_test.fit_transform(abcd)
print('done!')
# %%
""" DEFINE FEATURE SELECTOR """
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, select_features=True):
        self.feature_names = feature_names
        self.select_features = select_features

    def fit(self, X, y=None):
        X = X.copy()
        return self

    def transform(self, X):
        return X[self.feature_names]
# %%
""" DEFINE AVERAGE OVER HEMI """
class AverageHemi(BaseEstimator, TransformerMixin):
    num_in_hemi = 34

    def __init__(self, features, average_over_hemi=True):
        self._average_over_hemi = average_over_hemi
        self.new_cols = []
        self.avg_cort_tk = pd.DataFrame(data=None)
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.features]
        if self._average_over_hemi:
            for i in range(self.num_in_hemi):
                avg_tk = pd.DataFrame((X.iloc[:, i] + X.iloc[:, i+ self.num_in_hemi])/2)
                X = pd.concat([X, avg_tk], axis=1)
            X = X.iloc[:, len(self.features):]
            return X
# %%
""" DEFINE FEATURE DATA PIPELINE """
# This pipeline averages thickness values across cortex.
# If a continuous behaviour feature is also to be included as a predictor, then it should be identified and included
# here in the 'othernum_pipeline'. To use this, you have to also specify a list of other numerical predictors (e.g.
# continuous behavioral measures, age etc.

numpred_pipeline = Pipeline([
    ('selector_numpred', FeatureSelector(cort_cols, select_features=True)),
])

othernum_pipeline = Pipeline([
   ('selector_othernum', FeatureSelector(oth_predictors, select_features=True)),
])

feature_pipeline = FeatureUnion(transformer_list=[
    ('numpred_pipeline', numpred_pipeline),
    ('othernum_pipeline', othernum_pipeline)
])
# %%
# This preprocesses the categorical confounds (i.e. one hot encoding), and gathers the continuous confounds such as age etc.
catconfound_pipeline = Pipeline([
    ('select_cat', FeatureSelector(cat_confound_cols, select_features=True)),
    ('cat_onehot', OneHotEncoder(sparse_output=False, categories='auto', drop='first'))
])
numconfound_pipeline = Pipeline([
    ('selector_numconf', FeatureSelector(num_confound_cols, select_features=True)),
])

confound_pipeline = FeatureUnion(transformer_list=[
    ('numconfound_pipeline', numconfound_pipeline),
    ('cat_pipeline', catconfound_pipeline)
])
# %%
""" DEFINE REGRESSION """
# This defines the linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

def Regression(features, confounds):
    dict_features = {}
    for i in range(0, features.shape[1], 10):
        dict_features[i] = features.iloc[:, i:i+10]
    D = dict(enumerate(dict_features[x] for x in sorted(dict_features)))

    # Regress
    reg_results = []
    for i in range(len(D)):
        for col in range(len(list(D[i]))):
            y = pd.DataFrame(D[i]).iloc[:, col]
            m = lin_reg.fit(confounds, y)
            y_res = (y - m.predict(confounds))
            reg_results.append(y_res)

    reg_results = pd.DataFrame(reg_results).T
    return reg_results
# %%
""" APPLY PIPELINES AND REGRESSION """
print('Apply pipelines and regression...', end="")
data = [X_train, X_test]
reg_data = []
feat_data = []
con_data = []

for dataset in data:
    # Prepare features (brain and other continuous predictors)
    features_prepared = feature_pipeline.fit_transform(dataset)
    
    # Rename all features - include continuous features if needed
    features_prepared = pd.DataFrame(features_prepared, columns=[cort_cols + oth_predictors])
    feat_data.append(features_prepared)

    # Prepare confounds
    confounds_prepared = confound_pipeline.fit_transform(dataset)
    con_data.append(confounds_prepared)
    
    # Regression
    regressed = Regression(features_prepared, confounds_prepared)
    reg_data.append(regressed)
    reg_data[0]['group'] = y_train
    print(reg_data)
print('done!')
# %%
""" DEFINE CATEGORICAL PREDICTOR PIPELINE """
# This preprocesses the categorical predictors (i.e. one hot encoding)
# Be careful to include the appropriate number of column names
catpredictor_pipeline = Pipeline([
    ('select_cat', FeatureSelector(cat_predictors, select_features=True)),
    ('cat_onehot', OneHotEncoder(sparse_output=False, categories='auto'))
])

data = [X_train, X_test]
catpred_data = []

for dataset in data:
    catpred = catpredictor_pipeline.fit_transform(dataset)
    catpred = pd.DataFrame(catpred, columns=['p_edu_1', 'p_edu_2', 'p_edu_3', 'income_1', 'income_2', 'income_3', 'motor_1', 
                                             'motor_2', 'motor_3', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_3', 'speech_4', 'speech_5'])
    catpred_data.append(catpred)

cat_train = catpred_data[0]
cat_test = catpred_data[1]
print('cat_train shape', cat_train.shape)
print('cat_test shape', cat_test.shape)
# %%
""" STANDARD SCALING """
# Apply standard scaling to continuous features (i.e. brain data and any continuous predictors)
class StandardScaling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.self = self
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        cols = X.columns
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns = cols)
        return X
# %%
""" APPLY OUTLIER FILTERS """
# Identify outliers and convert to NaN (do not remove yet!)
class Outliers(BaseEstimator, TransformerMixin):
    def __init__(self, limit):
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for i in range(X.shape[1]):
            outliers = X.iloc[:, i][((X.iloc[:, i] < -self.limit) | (X.iloc[:, i] > self.limit))]
            for j in range(len(outliers)):
                X.iloc[:, i] = X.iloc[:, i].replace(outliers.iloc[j], np.NaN)
        return X
# %%
""" ADD IN LABEL DATA """
# Before we filter out NaNs we have to add in labels and other ***categorical one-hot encoded*** predictors
class AddInLabels(BaseEstimator, TransformerMixin):
    def __init__(self, y_train, y_test, cat_train, cat_test):
        self.y_train = y_train
        self.y_test = y_test
        self.cat_train = cat_train
        self.cat_test = cat_test
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # reset index to be able to match to regression data
        self.y_train = self.y_train.reset_index().group
        self.y_test = self.y_test.reset_index().group
        # add features to X
        if X.shape[0] == len(self.y_train):
            X['group'] = self.y_train
            X['p_edu_1'] = self.cat_train.p_edu_1
            X['p_edu_2'] = self.cat_train.p_edu_2
            X['p_edu_3'] = self.cat_train.p_edu_3
            X['income_1'] = self.cat_train.income_1
            X['income_2'] = self.cat_train.income_2
            X['income_3'] = self.cat_train.income_3
            X['motor_1'] = self.cat_train.motor_1
            X['motor_2'] = self.cat_train.motor_2
            X['motor_3'] = self.cat_train.motor_3
            X['motor_4'] = self.cat_train.motor_4
            X['motor_5'] = self.cat_train.motor_5
            X['speech_1'] = self.cat_train.speech_1
            X['speech_2'] = self.cat_train.speech_2
            X['speech_3'] = self.cat_train.speech_3
            X['speech_4'] = self.cat_train.speech_4
            X['speech_5'] = self.cat_train.speech_5
        else:
            X['group'] = self.y_test
            X['p_edu_1'] = self.cat_test.p_edu_1
            X['p_edu_2'] = self.cat_test.p_edu_2
            X['p_edu_3'] = self.cat_test.p_edu_3
            X['income_1'] = self.cat_test.income_1
            X['income_2'] = self.cat_test.income_2
            X['income_3'] = self.cat_test.income_3
            X['motor_1'] = self.cat_test.motor_1
            X['motor_2'] = self.cat_test.motor_2
            X['motor_3'] = self.cat_test.motor_3
            X['motor_4'] = self.cat_test.motor_4
            X['motor_5'] = self.cat_test.motor_5
            X['speech_1'] = self.cat_test.speech_1
            X['speech_2'] = self.cat_test.speech_2
            X['speech_3'] = self.cat_test.speech_3
            X['speech_4'] = self.cat_test.speech_4
            X['speech_5'] = self.cat_test.speech_5
        return X
# %%
""" FILTER NANs """
# Now we can remove NaNs
class DropNaN(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.self = self

    def fit(self, X, y=None):
        X = X.copy()
        return self

    def transform(self, X):
        return X.dropna().reset_index().drop(['index'], axis=1)
# %%
""" APPLY BOXCOX TRANSFORM """
# Apply BoxCox transform to data (brain, continuous predictors.)
class BoxCoxTransform(BaseEstimator, TransformerMixin):
    def __init__(self, boxcox=True):
        self.boxcox = boxcox

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Before BC transform, drop y labels
        X = X.copy()
        y = X.group
        a = X.p_edu_1
        b = X.p_edu_2
        c = X.p_edu_3
        d = X.income_1
        e = X.income_2
        f = X.income_3
        g = X.motor_1
        h = X.motor_2
        i = X.motor_3
        j = X.motor_4
        k = X.motor_5
        l = X.speech_1
        m = X.speech_2
        n = X.speech_3
        o = X.speech_4
        p = X.speech_5
        X = X.drop(['group', 'p_edu_1', 'p_edu_2', 'p_edu_3', 'income_1', 'income_2', 'income_3', 'motor_1', 
'motor_2', 'motor_3', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_3', 'speech_4', 'speech_5'], axis=1)
        cols = X.columns
        if self.boxcox:
            for i in range(X.shape[1]):
                con = (X.iloc[:, i] + (1 + np.absolute(np.min(X.iloc[:, i])))).astype('float')
                bc = scipy.stats.boxcox(con)[0]
                X = pd.concat([X, pd.Series(bc)], axis=1)
            X = X.iloc[:, len(cols):]
            X.columns = cols
            # Add back in categorical predictors and y labels
            X['group'] = y
            X['p_edu_1'] = a
            X['p_edu_2'] = b
            X['p_edu_3'] = c
            X['income_1'] = d
            X['income_2'] = e
            X['income_3'] = f
            X['motor_1'] = g
            X['motor_2'] = h
            X['motor_3'] = i
            X['motor_4'] = j
            X['motor_5'] = k
            X['speech_1'] = l
            X['speech_2'] = m
            X['speech_3'] = n
            X['speech_4'] = o
            X['speech_5'] = p

            return X
# %%
""" MANUAL STANDARD SCALING """
# BoxCox data may not be standard scaled
class ManualScaling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.self = self
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Before BC transform, drop categorical predictors and y labels
        X = X.copy()
        y = X.group
        a = X.p_edu_1
        b = X.p_edu_2
        c = X.p_edu_3
        d = X.income_1
        e = X.income_2
        f = X.income_3
        g = X.motor_1
        h = X.motor_2
        i = X.motor_3
        j = X.motor_4
        k = X.motor_5
        l = X.speech_1
        m = X.speech_2
        n = X.speech_3
        o = X.speech_4
        p = X.speech_5
        X = X.drop(['group', 'p_edu_1', 'p_edu_2', 'p_edu_3', 'income_1', 'income_2', 'income_3', 'motor_1', 
'motor_2', 'motor_3', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_3', 'speech_4', 'speech_5'], axis=1)
        cols = X.columns
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns = cols)
        # Add back in categorical predictors and y labels
        X['group'] = y
        X['p_edu_1'] = a
        X['p_edu_2'] = b
        X['p_edu_3'] = c
        X['income_1'] = d
        X['income_2'] = e
        X['income_3'] = f
        X['motor_1'] = g
        X['motor_2'] = h
        X['motor_3'] = i
        X['motor_4'] = j
        X['motor_5'] = k
        X['speech_1'] = l
        X['speech_2'] = m
        X['speech_3'] = n
        X['speech_4'] = o
        X['speech_5'] = p
        return X
# %%
""" APPLY STANDARD SCALER, OUTLIERS, NANs, BOXCOX TRANSFORM """

postreg_pipeline = Pipeline([
    ('standard_scaling', StandardScaling()),
    ('outliers', Outliers(limit=4)),
    ('addlabels', AddInLabels(y_train = y_train, y_test=y_test, cat_train = cat_train, cat_test=cat_test)),
    ('dropna', DropNaN()),
    ('boxcox', BoxCoxTransform(boxcox=True)),
    #('smote', SMOTESampling()),
    ('manual_scaling', ManualScaling())
])

prepared_data = []
for dataset in reg_data:
    new_data = postreg_pipeline.fit_transform(dataset)
    prepared_data.append(new_data)
# %%
X_train = prepared_data[0]
X_test = prepared_data[1]
#y_train = pd.Series(X_train.group)
y_train = pd.DataFrame(X_train['group']) 
X_train = prepared_data[0].drop(['group'], axis=1)
# %%
np.save("/Users/madeleineseitz/Desktop/X_train_all_data.npy", X_train)
np.save("/Users/madeleineseitz/Desktop/y_train_all_data.npy", y_train)
X_train = pd.DataFrame(X_train)
#X_train.columns = cols
X_train.to_csv('/Users/madeleineseitz/Desktop/X_train_all_data.csv', index=None, header=True)
y_train.to_csv('/Users/madeleineseitz/Desktop/y_train_all_data.csv', index=None, header=True)
#y_test = pd.Series(X_test.group)
y_test =  X_test[['group']]
y_test.to_csv('/Users/madeleineseitz/Desktop/y_test_all_data.csv', index=None, header=True)
y_test = y_test.to_numpy()
X_test = prepared_data[1].drop(['group'], axis=1)
np.save("/Users/madeleineseitz/Desktop/X_test_all_data.npy", X_test)
np.save("/Users/madeleineseitz/Desktop/y_test_all_data.npy", y_test)
X_test = pd.DataFrame(X_test)
#X_test.columns = cols
X_test.to_csv('/Users/madeleineseitz/Desktop/X_test_all_data.csv', index=None, header=True)
# %%
# selected_cols = ['tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crlg_nt', 
#  'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saigvcg_bscs_pulh', 'tfmri_saigvcg_bscs_purh', 
#  'tfmri_nback_all_208', 'tfmri_nback_all_215', 'smri_vol_cdk_paracnrh', 'nihtbx_flanker_agecorrected', 
#  'nihtbx_pattern_agecorrected', 'nihtbx_flanker_fc', 'nihtbx_pattern_fc', 'lmt_scr_avg_rt', 'lmt_scr_rt_correct',
# 'tfmri_ma_alrvn_b_scs_ayrh', 'tfmri_ma_rpvnfb_b_cds_clcgelh', 'tfmri_ma_rpvnfb_b_cds_roagelh', 'tfmri_sst_all_beh_incrlg_nt',
# 'tfmri_sacgvf_bscs_crbwmlh', 'tfmri_sacgvf_bscs_crbwmrh', 'tfmri_sacgvf_bscs_pdrh', 'tfmri_saigvcg_bscs_crbwmrh', 
# 'tfmri_saigvcg_bscs_crbcxrh', 'tfmri_nback_all_73', 'tfmri_nback_all_83', 'tfmri_nback_all_178', 'tfmri_nback_all_214', 
# 'tfmri_nback_all_223', 'tfmri_nback_all_232', 'tfmri_nback_all_262', 'tfmri_rec_all_beh_posf_dpr', 'tfmri_rec_all_beh_negface_pr', 'tfmri_rec_all_beh_negf_dp', 'smri_thick_cdk_cuneuslh', 'smri_thick_cdk_rracatelh',
# 'smri_thick_cdk_rrmdfrlh', 'smri_thick_cdk_sufrlh', 'smri_thick_cdk_supllh', 'smri_thick_cdk_trvtmlh', 'smri_thick_cdk_parsopcrh', 'smri_thick_cdk_sufrrh', 'smri_vol_cdk_trvtmlh', 'smri_vol_cdk_iftmrh',
# 'sit_scr_values_count3', 'lmt_scr_num_timed_out', 'tfmri_mid_all_beh_hrwnfb_nt', 'tfmri_mid_all_beh_hlnfb_nt', 
# 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sacgvf_bscs_pulh', 'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_sst_nbeh_nruns', 'tfmri_sst_all_beh_crgo_nt', 'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sacgvf_bscs_purh', 'tfmri_sacsvcg_bscs_crbcxlh', 
# 'tfmri_saasvcg_bscs_tplh', 'tfmri_saasvcg_bscs_aalh', 'tfmri_sacsvis_bscs_aarh', 'tfmri_nback_all_148',
# 'tfmri_nback_all_154', 'tfmri_nback_all_185', 'tfmri_nback_all_216', 'tfmri_nback_all_224', 'tfmri_nback_all_237',
# 'tfmri_rec_all_beh_newnf_hr', 'smri_thick_cdk_fusiformlh', 'smri_thick_cdk_iftmlh', 'smri_thick_cdk_parsobislh',
# 'smri_thick_cdk_pclh', 'smri_thick_cdk_smlh', 'smri_thick_cdk_frpolelh', 'smri_thick_cdk_banksstsrh',
# 'smri_thick_cdk_ifplrh', 'smri_thick_cdk_insularh', 'smri_vol_cdk_rrmdfrrh', 'sit_scr_expr_mratdiff1',
# 'tfmri_ma_alvcr_b_scs_tprh', 'tfmri_sst_all_beh_s_nt', 'tfmri_nback_all_51', 'tfmri_nback_all_186', 'smri_thick_cdk_ptcaterh', 
# 'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum', 'peq_ss_overt_aggression', 'peq_ss_relational_aggs', 'peq_ss_overt_victim',
# 'total_substances_used', 'weeks_preterm', 'injury_totals', 'p_edu_1', 'p_edu_2', 'p_edu_3', 'income_1', 'income_2', 'income_3',
# 'motor_1', 'motor_2', 'motor_3', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_3', 'speech_4', 'speech_5']

# X_train_reduced = X_train[selected_cols]
# X_test_reduced = X_test[selected_cols]
# X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_reduced.csv')
# X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_reduced.csv')
# %%
