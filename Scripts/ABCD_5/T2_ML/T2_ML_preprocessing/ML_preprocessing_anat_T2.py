#This script preprocesses anatomical data that will be fed into the Scikit-Learn ML models (uniform data)
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
#Structural data
anat = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/uniform_csvs/T2/abcd5_uniform_T2_anat_abcd_nonan.csv')
anat = pd.DataFrame(anat)
anat = anat.drop(['Unnamed: 0'], axis=1)
abcd = anat
#abcd = abcd.drop(columns= ['Unnamed: 0'])
# %%
""" DEFINE FEATURE CATEGORIES """
# This script is set up as follows:
# 1. Split data in to train / test
# 2. Identify brain data (including average across hemispheres). At this point, if you want to include a continuous
# behavior predictor ('num_predictors'), identify it separately and include it in the Feature Data Pipeline
# 2. Regress confounds on to brain data.
num_confound_cols = ['interview_age']
cat_confound_cols = ['sex', 'p_edu', 'income', 'race']

# Predictive Features
cort_cols = list(abcd)[2:-6]
#dropnans
abcd = abcd.dropna() #shape (1370, 504) HR n = 658
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
        X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(features, label, X.src_subject_id, test_size=self.test_size, random_state=42)
        self.X_train = X_train.drop(['src_subject_id'], axis=1)
        self.X_test = X_test.drop(['src_subject_id'], axis=1)
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

#othernum_pipeline = Pipeline([
#    ('selector_othernum', FeatureSelector(oth_predictors, select_features=True)),
#])

feature_pipeline = FeatureUnion(transformer_list=[
    ('numpred_pipeline', numpred_pipeline)
    #('othernum_pipeline', othernum_pipeline),
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
    features_prepared = pd.DataFrame(features_prepared, columns=[cort_cols])
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
    def __init__(self, y_train, y_test):
        self.y_train = y_train
        self.y_test = y_test
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # reset index to be able to match to regression data
        self.y_train = self.y_train.reset_index().group
        self.y_test = self.y_test.reset_index().group
        # add features to X
        if X.shape[0] == len(self.y_train):
            X['group'] = self.y_train
        else:
            X['group'] = self.y_test
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
        X = X.drop(['group'], axis=1)
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
        X = X.drop(['group'], axis=1)
        cols = X.columns
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns = cols)
        # Add back in categorical predictors and y labels
        X['group'] = y
        return X
# %%
""" APPLY STANDARD SCALER, OUTLIERS, NANs, BOXCOX TRANSFORM """

postreg_pipeline = Pipeline([
    ('standard_scaling', StandardScaling()),
    ('outliers', Outliers(limit=4)),
    ('addlabels', AddInLabels(y_train = y_train, y_test=y_test)),
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
np.save("/Users/madeleineseitz/Desktop/X_train_anat_data_abcd5_T2.npy", X_train)
np.save("/Users/madeleineseitz/Desktop/y_train_anat_data_abcd5_T2.npy", y_train)
X_train = pd.DataFrame(X_train)
#X_train.columns = cols
X_train.to_csv('/Users/madeleineseitz/Desktop/X_train_anat_data_abcd5_T2.csv', index=None, header=True)
y_train.to_csv('/Users/madeleineseitz/Desktop/y_train_anat_data_abcd5_T2.csv', index=None, header=True)
#y_test = pd.Series(X_test.group)
y_test =  X_test[['group']]
y_test.to_csv('/Users/madeleineseitz/Desktop/y_test_anat_data_abcd5_T2.csv', index=None, header=True)
y_test = y_test.to_numpy()
np.save("/Users/madeleineseitz/Desktop/y_test_anat_data_abcd5_T2.npy", y_test)
X_test = prepared_data[1].drop(['group'], axis=1)
np.save("/Users/madeleineseitz/Desktop/X_test_anat_data_abcd5_T2.npy", X_test)
X_test = pd.DataFrame(X_test)
#X_test.columns = cols
X_test.to_csv('/Users/madeleineseitz/Desktop/X_test_anat_data_abcd5_T2.csv', index=None, header=True)
# %%
# %%
no_doubles = ['smri_thick_cdk_cdacatelh', 'smri_thick_cdk_cuneuslh', 'smri_thick_cdk_ihcatelh', 'smri_thick_cdk_paracnlh',
  'smri_thick_cdk_postcnlh', 'smri_thick_cdk_precnlh', 'smri_thick_cdk_rracatelh', 'smri_thick_cdk_sufrlh',
 'smri_thick_cdk_ehinalrh', 'smri_thick_cdk_iftmrh', 'smri_thick_cdk_loccrh', 'smri_thick_cdk_paracnrh',
'smri_thick_cdk_periccrh', 'smri_thick_cdk_rrmdfrrh', 'smri_thick_cdk_sutmrh', 'smri_thick_cdk_mean',
 'smri_vol_cdk_banksstslh', 'smri_vol_cdk_cdacatelh', 'smri_vol_cdk_cdmdfrlh', 'smri_vol_cdk_fusiformlh',
'smri_vol_cdk_iftmlh', 'smri_vol_cdk_mobfrlh', 'smri_vol_cdk_paracnlh', 'smri_vol_cdk_postcnlh', 'smri_vol_cdk_ptcatelh',
'smri_vol_cdk_precnlh', 'smri_vol_cdk_fusiformrh', 'smri_vol_cdk_lobfrrh', 'smri_vol_cdk_lingualrh', 'smri_vol_cdk_mdtmrh',
'smri_vol_cdk_precnrh', 'smri_vol_cdk_pcrh', 'smri_vol_cdk_suplrh', 'smri_vol_cdk_smrh', 'smri_vol_cdk_tmpolerh',
 'smri_vol_cdk_trvtmrh', 'smri_vol_cdk_totallh', 'smri_thick_cdk_fusiformlh', 'smri_thick_cdk_mobfrlh',
 'smri_thick_cdk_fusiformrh', 'smri_vol_cdk_parsopclh', 'smri_vol_cdk_paracnrh', 'smri_vol_cdk_sufrrh',
'smri_thick_cdk_pericclh', 'smri_thick_cdk_lobfrrh', 'smri_vol_cdk_linguallh', 'smri_vol_cdk_pericclh', 
'smri_vol_cdk_sufrlh', 'smri_vol_cdk_banksstsrh', 'smri_vol_cdk_total', 'smri_thick_cdk_cdmdfrlh', 'smri_thick_cdk_ifpllh',
 'smri_thick_cdk_parahpallh', 'smri_thick_cdk_ptcatelh', 'smri_thick_cdk_supllh', 'smri_thick_cdk_banksstsrh', 
 'smri_thick_cdk_smrh', 'smri_thick_cdk_insularh', 'smri_vol_cdk_smlh', 'smri_vol_cdk_parahpalrh', 'smri_vol_cdk_insularh']
# %%
X_train_reduced = X_train[no_doubles]
X_test_reduced = X_test[no_doubles]
# %%
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_anat_reduced_abcd5_T2.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_anat_reduced_abcd5_T2.csv')
# %%
