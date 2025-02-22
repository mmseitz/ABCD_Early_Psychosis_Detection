#This script preprocesses environmental data that will be fed into the Scikit-Learn ML models 
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
#### Import T2 raw data
print('Importing raw data...', end="")
abcd = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/uniform_csvs/T2/abcd5_uniform_T2_env_abcd_nonan.csv')
abcd = abcd.drop(columns= ['Unnamed: 0'])
# %%
#perform train test split on T2 data, maintianing src_subject_id
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
        X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(features, label, X.src_subject_id, test_size=self.test_size, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_ind = train_ind
        self.test_ind = test_ind
        return self.X_train, self.X_test, self.y_train, self.y_test, self.train_ind, self.test_ind

train_test = TrainTestSplit('group', 0.2)
X_train, X_test, y_train, y_test, train_ind, test_ind = train_test.fit_transform(abcd)
print('done!')
# %%
#save ids present in abcd T2 train set
ids = list(X_train['src_subject_id'])
# %%
#### Import T4 raw data
print('Importing raw data...', end="")
abcd = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/uniform_csvs/T4/abcd5_uniform_T4_env_abcd_nonan.csv')
abcd = abcd.drop(columns= ['Unnamed: 0'])
# %%
#calculate how many ids in T4 are not in T2 train set
ok_list = []
T4_ids = list(abcd['src_subject_id'])
for i in T4_ids: 
    if i not in ids:
        ok_list.append(i)
print(len(ok_list))
# %%
abcd['doubled'] = 0
for row in abcd.index: 
    if abcd['src_subject_id'][row] not in ok_list:
        abcd['doubled'][row] = 1
abcd = abcd.loc[abcd['doubled'] == 0]
abcd['group'].sum()
X_test = abcd
y_test = abcd['group']
#abcd shape (139, 46)
# abcd T4 HR = 80
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
cat_predictors = ['p_edu', 'income', 'motor_dev_comparison', 'speech_dev_comparison', 'preterm', 'dim_yesno_q1', 'dim_yesno_q2', 
                  'dim_yesno_q3', 'dim_yesno_q4', 'neighborhood_crime_y']

oth_predictors = ['total_substances_used', 'weeks_preterm', 'age_roll_over', 'age_sit', 'age_walk',
                  'age_first_word', 'total_times_to_ER', 'times_to_ER_past_year', 'injury_totals', 'ple_p_ss_total_number', 
                  'ple_p_ss_total_good', 'ple_p_ss_total_bad', 'ple_p_ss_affected_good_sum', 'ple_p_ss_affected_bad_sum', 
                  'ple_y_ss_total_number', 'ple_y_ss_total_good', 'ple_y_ss_total_bad', 'ple_y_ss_affected_good_sum',
                  'ple_y_ss_affected_bad_sum', 'peq_ss_relational_victim', 'peq_ss_reputation_aggs', 
                  'peq_ss_reputation_victim', 'peq_ss_overt_aggression', 'peq_ss_overt_victim', 'peq_ss_relational_aggs',
                  'dim_y_ss_mean', 'comc_ss_cohesion_p', 'comc_ss_control_p', 'comc_ss_collective_capacity_p']
# %%
#note shape
abcd.shape #shape (204, 45) HR n = 106
abcd['group'].sum()
# %%
# """ DEFINE AND APPLY TRAIN/TEST SPLIT """
# print('Define train/test split...', end="")
# class TrainTestSplit(BaseEstimator, TransformerMixin):
#     def __init__(self, label, test_size):
#         self.test_size = test_size
#         self.label = label
#         self.X_train = pd.DataFrame(data=None)
#         self.X_test = pd.DataFrame(data=None)
#         self.y_train = pd.DataFrame(data=None)
#         self.y_test = pd.DataFrame(data=None)
#         self.train_ind = pd.DataFrame(data=None)
#         self.test_ind = pd.DataFrame(data=None)

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         label = X[self.label]
#         features = X.drop([self.label], axis=1)
#         X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(features, label, X.src_subject_id, test_size=self.test_size, random_state=42)
#         self.X_train = X_train.drop(['src_subject_id'], axis=1)
#         self.X_test = X_test.drop(['src_subject_id'], axis=1)
#         self.y_train = y_train
#         self.y_test = y_test
#         self.train_ind = train_ind
#         self.test_ind = test_ind
#         return self.X_train, self.X_test, self.y_train, self.y_test, self.train_ind, self.test_ind

# train_test = TrainTestSplit('group', 0.2)
# X_train, X_test, y_train, y_test, train_ind, test_ind = train_test.fit_transform(abcd)
# print('done!')
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

# numpred_pipeline = Pipeline([
#     ('selector_numpred', FeatureSelector(cort_cols, select_features=True)),
# ])

othernum_pipeline = Pipeline([
   ('selector_othernum', FeatureSelector(oth_predictors, select_features=True)),
])

feature_pipeline = FeatureUnion(transformer_list=[
    #('numpred_pipeline', numpred_pipeline)
    ('othernum_pipeline', othernum_pipeline),
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
data = [X_test]
reg_data = []
feat_data = []
con_data = []

for dataset in data:
    # Prepare features (brain and other continuous predictors)
    features_prepared = feature_pipeline.fit_transform(dataset)
    
    # Rename all features - include continuous features if needed
    features_prepared = pd.DataFrame(features_prepared, columns=[oth_predictors])
    feat_data.append(features_prepared)

    # Prepare confounds
    confounds_prepared = confound_pipeline.fit_transform(dataset)
    con_data.append(confounds_prepared)
    
    # Regression
    regressed = Regression(features_prepared, confounds_prepared)
    reg_data.append(regressed)
    reg_data[0]['group'] = y_test
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

data = [X_test]
catpred_data = []

for dataset in data:
    catpred = catpredictor_pipeline.fit_transform(dataset)
    catpred = pd.DataFrame(catpred, columns=['p_edu_1', 'p_edu_2', 'p_edu_3', 'income_1', 'income_2', 'income_3', 'motor_1', 
                                             'motor_2', 'motor_3', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_3', 'speech_4', 'speech_5', 
                                             'preterm_1', 'preterm_2', 'disc_1_a', 'disc_1_b', 'disc_2_a', 'disc_2_b', 'disc_3_a', 
                                             'disc_3_b', 'disc_4_a', 'disc_4_b', 'crime_1', 'crime_2', 'crime_3', 'crime_4', 
                                             'crime_5'])
    catpred_data.append(catpred)

cat_test = catpred_data[0]
#cat_test = catpred_data[1]
#print('cat_train shape', cat_train.shape)
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
    def __init__(self, y_test, cat_test):
        #self.y_train = y_train
        self.y_test = y_test
        #self.cat_train = cat_train
        self.cat_test = cat_test
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # reset index to be able to match to regression data
        #self.y_train = self.y_train.reset_index().group
        self.y_test = self.y_test.reset_index().group
        # add features to X
        if X.shape[0] == len(self.y_test):
            # X['group'] = self.y_train
            # X['p_edu_1'] = self.cat_train.p_edu_1
            # X['p_edu_2'] = self.cat_train.p_edu_2
        #     X['p_edu_3'] = self.cat_train.p_edu_3
        #     X['income_1'] = self.cat_train.income_1
        #     X['income_2'] = self.cat_train.income_2
        #     X['income_3'] = self.cat_train.income_3
        #     X['motor_1'] = self.cat_train.motor_1
        #     X['motor_2'] = self.cat_train.motor_2
        #     X['motor_3'] = self.cat_train.motor_3
        #     X['motor_4'] = self.cat_train.motor_4
        #     X['motor_5'] = self.cat_train.motor_5
        #     X['speech_1'] = self.cat_train.speech_1
        #     X['speech_2'] = self.cat_train.speech_2
        #     X['speech_3'] = self.cat_train.speech_3
        #     X['speech_4'] = self.cat_train.speech_4
        #     X['speech_5'] = self.cat_train.speech_5
        #     X['preterm_1'] = self.cat_train.preterm_1
        #     X['preterm_2'] = self.cat_train.preterm_2
        #     X['disc_1_a'] = self.cat_train.disc_1_a
        #     X['disc_1_b'] = self.cat_train.disc_1_b
        #     X['disc_2_a'] = self.cat_train.disc_2_a
        #     X['disc_2_b'] = self.cat_train.disc_2_b
        #     X['disc_3_a'] = self.cat_train.disc_3_a
        #     X['disc_3_b'] = self.cat_train.disc_3_b
        #     X['disc_4_a'] = self.cat_train.disc_4_a
        #     X['disc_4_b'] = self.cat_train.disc_4_b
        #     X['crime_1'] = self.cat_train.crime_1
        #     X['crime_2'] = self.cat_train.crime_2
        #     X['crime_3'] = self.cat_train.crime_3
        #     X['crime_4'] = self.cat_train.crime_4
        #     X['crime_5'] = self.cat_train.crime_5
        # else:
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
            X['preterm_1'] = self.cat_test.preterm_1
            X['preterm_2'] = self.cat_test.preterm_2
            X['disc_1_a'] = self.cat_test.disc_1_a
            X['disc_1_b'] = self.cat_test.disc_1_b
            X['disc_2_a'] = self.cat_test.disc_2_a
            X['disc_2_b'] = self.cat_test.disc_2_b
            X['disc_3_a'] = self.cat_test.disc_3_a
            X['disc_3_b'] = self.cat_test.disc_3_b
            X['disc_4_a'] = self.cat_test.disc_4_a
            X['disc_4_b'] = self.cat_test.disc_4_b
            X['crime_1'] = self.cat_test.crime_1
            X['crime_2'] = self.cat_test.crime_2
            X['crime_3'] = self.cat_test.crime_3
            X['crime_4'] = self.cat_test.crime_4
            X['crime_5'] = self.cat_test.crime_5
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
        q = X.preterm_1
        r = X.preterm_2
        s = X.disc_1_a
        t = X.disc_1_b
        u = X.disc_2_a
        v = X.disc_2_b
        w = X.disc_3_a
        aa = X.disc_3_b
        bb = X.disc_4_a
        cc = X.disc_4_b
        dd = X.crime_1
        ee = X.crime_2
        ff = X.crime_3
        gg = X.crime_4
        hh = X.crime_5
        X = X.drop(['group', 'p_edu_1', 'p_edu_2', 'p_edu_3', 'income_1', 'income_2', 'income_3', 'motor_1', 
'motor_2', 'motor_3', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_3', 'speech_4', 'speech_5', 
'preterm_1', 'preterm_2', 'disc_1_a', 'disc_1_b', 'disc_2_a', 'disc_2_b', 'disc_3_a', 'disc_3_b', 'disc_4_a',
'disc_4_b', 'crime_1', 'crime_2', 'crime_3', 'crime_4', 'crime_5'], axis=1)
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
            X['preterm_1'] = q
            X['preterm_2'] = r
            X['disc_1_a'] = s
            X['disc_1_b'] = t
            X['disc_2_a'] = u
            X['disc_2_b'] = v
            X['disc_3_a'] = w
            X['disc_3_b'] = aa
            X['disc_4_a'] = bb
            X['disc_4_b'] = cc
            X['crime_1'] = dd
            X['crime_2'] = ee
            X['crime_3'] = ff
            X['crime_4'] = gg
            X['crime_5'] = hh

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
        q = X.preterm_1
        r = X.preterm_2
        s = X.disc_1_a
        t = X.disc_1_b
        u = X.disc_2_a
        v = X.disc_2_b
        w = X.disc_3_a
        aa = X.disc_3_b
        bb = X.disc_4_a
        cc = X.disc_4_b
        dd = X.crime_1
        ee = X.crime_2
        ff = X.crime_3
        gg = X.crime_4
        hh = X.crime_5
        X = X.drop(['group', 'p_edu_1', 'p_edu_2', 'p_edu_3', 'income_1', 'income_2', 'income_3', 'motor_1', 
'motor_2', 'motor_3', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_3', 'speech_4', 'speech_5', 
'preterm_1', 'preterm_2', 'disc_1_a', 'disc_1_b', 'disc_2_a', 'disc_2_b', 'disc_3_a', 'disc_3_b', 'disc_4_a',
'disc_4_b', 'crime_1', 'crime_2', 'crime_3', 'crime_4', 'crime_5'], axis=1)
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
        X['preterm_1'] = q
        X['preterm_2'] = r
        X['disc_1_a'] = s
        X['disc_1_b'] = t
        X['disc_2_a'] = u
        X['disc_2_b'] = v
        X['disc_3_a'] = w
        X['disc_3_b'] = aa
        X['disc_4_a'] = bb
        X['disc_4_b'] = cc
        X['crime_1'] = dd
        X['crime_2'] = ee
        X['crime_3'] = ff
        X['crime_4'] = gg
        X['crime_5'] = hh
        return X
# %%
""" APPLY STANDARD SCALER, OUTLIERS, NANs, BOXCOX TRANSFORM """

postreg_pipeline = Pipeline([
    ('standard_scaling', StandardScaling()),
    ('outliers', Outliers(limit=4)),
    ('addlabels', AddInLabels(y_test=y_test, cat_test=cat_test)),
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
# np.save("/Users/madeleineseitz/Desktop/X_train_env_data_abcd5_T4.npy", X_train)
# np.save("/Users/madeleineseitz/Desktop/y_train_env_data_abcd5_T4.npy", y_train)
# X_train = pd.DataFrame(X_train)
#X_train.columns = cols
# X_train.to_csv('/Users/madeleineseitz/Desktop/X_train_env_data_abcd5_T4.csv', index=None, header=True)
# y_train.to_csv('/Users/madeleineseitz/Desktop/y_train_env_data_abcd5_T4.csv', index=None, header=True)
#y_test = pd.Series(X_test.group)
X_test = prepared_data[0]
y_test =  X_test[['group']]
y_test.to_csv('/Users/madeleineseitz/Desktop/y_test_env_data_abcd5_T4_clean.csv', index=None, header=True)
y_test = y_test.to_numpy()
X_test = X_test.drop(['group'], axis=1)
np.save("/Users/madeleineseitz/Desktop/X_test_env_data_abcd5_T4_clean.npy", X_test)
np.save("/Users/madeleineseitz/Desktop/y_test_env_data_abcd5_T4_clean.npy", y_test)
X_test = pd.DataFrame(X_test)
#X_test.columns = cols
X_test.to_csv('/Users/madeleineseitz/Desktop/X_test_env_data_abcd5_T4.csv', index=None, header=True)
# %%
no_doubles = ['weeks_preterm', 'age_roll_over', 'age_walk', 'total_times_to_ER', 'ple_p_ss_affected_good_sum', 
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
#X_train_reduced = X_train[no_doubles]
X_test_reduced = X_test[no_doubles]
# %%
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_env_reduced_abcd5_T4_clean.csv')
#X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_env_reduced_abcd5_T4.csv')
# %%
print(no_doubles)
# %%
['weeks_preterm', 'age_roll_over', 'age_walk', 'total_times_to_ER', 'ple_p_ss_affected_good_sum', 
 'ple_y_ss_total_number', 'ple_y_ss_total_good', 'ple_y_ss_affected_good_sum',
 'ple_y_ss_affected_bad_sum', 'peq_ss_relational_victim', 'peq_ss_reputation_victim',
   'peq_ss_overt_victim', 'dim_y_ss_mean', 'comc_ss_cohesion_p', 'comc_ss_control_p',
 'comc_ss_collective_capacity_p', 'p_edu_2', 'p_edu_3', 'income_1', 'motor_1', 'motor_3',
'speech_1', 'speech_4', 'speech_5', 'preterm_2', 'disc_2_a', 'disc_2_b',
 'disc_3_a', 'disc_3_b', 'crime_2', 'crime_3', 'age_first_word', 
'ple_p_ss_total_number', 'ple_p_ss_total_good', 'ple_p_ss_total_bad', 'ple_y_ss_total_bad',
 'peq_ss_overt_aggression', 'peq_ss_relational_aggs', 'p_edu_1', 'income_2', 'speech_2',
 'preterm_1', 'disc_1_a', 'disc_1_b', 'crime_4', 'crime_5', 'times_to_ER_past_year',
   'ple_p_ss_affected_bad_sum', 'peq_ss_reputation_aggs', 'income_3', 'motor_2', 
   'motor_4', 'disc_4_a', 'disc_4_b']
# %%
len(no_doubles)

# %%
