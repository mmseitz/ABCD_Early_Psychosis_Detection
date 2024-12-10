#Full model (cognitive features only)
# %%
print('Importing modules...', end='')
import warnings
#warnings.filterwarnings('ignore', category=DeprecationWarning)
#warnings.filterwarnings('ignore', category=FutureWarning)
#warnings.filterwarnings('ignore', category=UserWarning)
#warnings.filterwarnings('ignore', 'Solver terminated early.*')
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("Warning not shown")


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

from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import recall_score, roc_auc_score, precision_recall_fscore_support, cohen_kappa_score, fbeta_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, balanced_accuracy_score, log_loss, precision_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
from scipy.stats import binomtest
from sklearn.inspection import permutation_importance

print('done!')
# %%
# %%
# Where to save figures
PROJECT_ROOT_DIR = '/Users/madeleineseitz/Desktop/'
CHAPTER_ID = 'sklearn_uniform_cog_only_ABCD5_T2'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, CHAPTER_ID, 'Images')
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_pdf(fig_id, tight_layout=True, fig_extension='pdf', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def save_png(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
# Set directory
os.chdir("/Users/madeleineseitz/Desktop/sklearn_uniform_cog_only_ABCD5_T2")
# %%
# Import data....
def data():
    print("Loading data...", end="")
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_cog_only/X_train_uniform_cog_reduced_abcd5_T2.csv')
    X_train = X_train.drop(columns= ['Unnamed: 0'])
    X_train = X_train.to_numpy()
    y_train = pd.read_csv("/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_cog_only/y_train_cog_only_data_abcd5_T2.csv")
    y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_cog_only/X_test_uniform_cog_reduced_abcd5_T2.csv')
    X_test = X_test.drop(columns= ['Unnamed: 0'])
    X_test = X_test.to_numpy()
    y_test = pd.read_csv("/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_cog_only/y_test_cog_only_data_abcd5_T2.csv")
    y_test = y_test['group'].to_numpy()
    print("done!")
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = data()
# %%
names = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/T2_abcd5_cog_only/X_train_uniform_cog_reduced_abcd5_T2.csv')
names = names.drop(columns= ['Unnamed: 0'])
feature_names = names.columns
# %%
y_test.shape
# %%
# Define logloss threshold....
prob_low = Counter(y_test)[0] / y_test.shape[0]
y_low = Counter(y_test)[0]
log_low = np.log(prob_low)  # np.log is ln!
prob_high = Counter(y_test)[1] / y_test.shape[0]
y_high = Counter(y_test)[1]
log_high = np.log(prob_high)
LogLoss = -(y_low*log_low + y_high*log_high)
logloss_threshold = LogLoss / y_test.shape[0]

print(log_low)
print(log_high)
print('LogLoss threshold', round(logloss_threshold,3))
# %%
# Define Metrics....

def get_label_results(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    acc = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # how many obese properly classified (same as recall)
    specificity = tn / (fp + tn)  # how many lean people correctly classified
    recall = recall_score(y_true, y_pred)  # fraction obese properly classified
    precision = precision_score(y_true, y_pred)  # fraction of error in obese classifications
    cohenkappa = cohen_kappa_score(y_true, y_pred)  # how much better model is over random classifier
    # based on class frequencies
    f1 = f1_score(y_true, y_pred)

    # p-values
    success = r * s
    success = success.astype('int')
    prob_class1 = np.count_nonzero(y_true) / float(len(y_true))
    pval_bacc_result = binomtest(success[0] + success[1], s[0] + s[1], p=0.5, alternative='greater')
    pval_bacc = pval_bacc_result.pvalue
    pval_low_result = binomtest(success[0], s[0], 1 - prob_class1, alternative='greater')
    pval_low = pval_low_result.pvalue
    pval_high_result = binomtest(success[1], s[1], prob_class1, alternative='greater')
    pval_high = pval_high_result.pvalue

    print('Results')
    print('Accuracy = %.2f' % acc, '(p-val', round(pval_bacc, 3), ')')
    print("Specificity = %.2f " % specificity, '(p-val %.3f' % pval_low, ')', '(true lean)')
    print('Recall (Sensitiviy) = %.2f' % recall, '(p-val %.3f' % pval_high, ')', '(true obese)')
    print('Precision = %.2f' % precision)
    print('F1 score = %.2f' % f1)
    print('Cohen-Kappa = %.2f' % cohenkappa)
    print('tn, fp, fn, tp', tn,  fp, fn, tp)

    return sensitivity, specificity, acc, recall, precision, f1, cohenkappa

def get_prob_results(y_true, y_proba):
    logloss = log_loss(y_true, y_proba)
    rocauc = roc_auc_score(y_true, y_proba[:, 1])
    brier = brier_score_loss(y_true, y_proba)

    print('Log loss = %.3f' % logloss)
    print('ROC AUC = %.3f' % rocauc)
    print('Brier score = %.2f' % brier)

    return logloss, rocauc

metrics = ['neg_log_loss', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# %%
# Def SelectColumns....
class SelectColumns():
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, **transform_params):
        copy_df = X[self.columns].copy()
        return copy_df

    def fit(self, X, y=None, **fit_params):
        return self

    def __repr__(self):
        return "<__main__.SelectColumns = " + str(self.columns) + ">"
# %%
# Def plot_histogram....
def plot_histogram(estimator, name, y_test, y_prob):
    if hasattr(estimator, 'predict_proba'):
        low = []
        high = []
        for i in range(len(y_test)):
            if np.array(y_test)[i] == 0:
                low.append(y_prob[:, 1][i])
            elif np.array(y_test)[i] == 1:
                high.append(y_prob[:, 1][i])

        low = np.array(low).ravel()
        high = np.array(high).ravel()

        sns.distplot(low, hist=True, kde=True, bins=20, color='darkblue', label='low',
             hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 1})
        sns.distplot(high, hist=True, kde=True, bins=20, color='green', label='high',
             hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 1})
        plt.legend(loc='best')
        save_pdf(name)
        save_png(name)
        plt.show()
# %%
# Def get_results....
def get_results(model, estimator, X_train, y_train, X_test, y_test):
    # Fit on data
    best = estimator.fit(X_train, y_train)
    best.best_estimator_

    # Apply final model
    final_estimator = best.best_estimator_
    final_model = final_estimator.fit(X_train, y_train.ravel())

    # Test Class Predictions
    #y_test_predict = final_model.predict(X_test)
    #auc = get_label_results(y_test, y_test_predict)

    if hasattr(estimator, 'predict_proba'):
        Results = pd.DataFrame(best.cv_results_)
        name = str(model) + '_RCV' + '.csv'
        Results.to_csv(name, index=None, header=True)
        Results.sort_values(by=['mean_test_neg_log_loss'], ascending=False)[
        ['rank_test_neg_log_loss', 'mean_test_roc_auc', 'mean_test_recall',
         'mean_test_precision', 'mean_test_f1']].head(5)

        # Test Probability Predictions
        y_prob = final_model.predict_proba(X_test)
        #logloss = get_prob_results(y_test, y_prob[:, 1])

        # Plot and save histogram
        name = str(model) + '_histogram_noSMOTE'
        plot_histogram(estimator, name, y_test, y_prob)
    else:
        Results = pd.DataFrame(best.cv_results_)
        Results = Results.sort_values(['rank_test_neg_log_loss'])
        name = str(model) + '_RCV' + '.csv'
        Results.to_csv(name, index=None, header=True)
        Results.sort_values(by=['mean_test_roc_auc'], ascending=False)[
            ['mean_test_roc_auc', 'mean_test_recall',
             'mean_test_precision', 'mean_test_f1']].head(5)

    # Save model
    name = str(model) + '_RCV_final_model' + '.sav'
    joblib.dump(final_model, name)

    return final_estimator

# %%
# Def plot_learning_curve....
from sklearn.model_selection import ShuffleSplit, learning_curve

def plot_learning_curve(estimator, title,  X, y, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 20)):

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring='neg_log_loss')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="darkblue",
                 label="training loss", linewidth=1)
    plt.plot(train_sizes, test_scores_mean, 'o-', color="green",
                 label="cross-validation loss", linewidth=1)
    plt.ylim(-1, -0.2)
    plt.hlines(-logloss_threshold, 300, 4000, colors='violet', label='chance level', linestyles='dashed', linewidth=1)
    plt.hlines(-base_logloss, 300, 4000, colors='orange', label='base logloss', linestyles='dashed', linewidth=1)
    plt.legend(loc="best")
    plt.title(title)
    fig_id = title + '_learningcurves'
    save_pdf(fig_id)
    save_png(fig_id)
    plt.show()

    return plt
# %%
# Feature Selection....
feature_grid = {'feature_eng': [None,
                              PCA(n_components=0.95),
                              SelectColumns(['tfmri_ma_aclvn_b_scs_aylh', 'tfmri_ma_acmvn_b_scs_aylh', 'tfmri_ma_rpvnfb_b_cds_clmdfrlh', 'tfmri_ma_rpvnfb_b_cds_romdfrrh', 
 'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_sacsvcg_bscs_crbwmrh', 'tfmri_sacsvcg_bscs_pdrh',
   'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_crbcxrh', 'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_incrgo_nt', 'tfmri_sst_all_beh_incrlg_nt', 'tfmri_sst_all_beh_nrgo_rt', 
 'tfmri_sst_all_beh_incrs_nt', 'tfmri_nback_all_4', 'tfmri_nback_all_52', 'tfmri_nback_all_65',
 'tfmri_nback_all_103', 'tfmri_nback_all_111', 'tfmri_nback_all_185', 'tfmri_nback_all_215', 'tfmri_nback_all_254',
 'tfmri_nback_all_262', 'tfmri_rec_all_beh_neutface_br', 'tfmri_rec_all_beh_negface_pr', 'gdt_scr_values_account_balance', 
 'sit_scr_expr_mrt_finalrat', 'sit_scr_values_count3', 'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc', 'lmt_scr_num_timed_out',
 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_rpvnfb_b_cds_clcgelh', 'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_ma_lvnfb_b_cds_roagelh',
 'tfmri_mid_all_beh_srwnfb_nt', 'tfmri_mid_all_beh_ntpfb_rate', 'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbwmlh', 
 'tfmri_sacsvcg_bscs_tprh', 'tfmri_saisvcg_bscs_crbwmrh', 'tfmri_saisvcg_bscs_cderh', 'tfmri_sacsvis_bscs_crbwmlh',
 'tfmri_sst_all_beh_s_nt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_incrgo_mrt', 'tfmri_sst_all_beh_incrs_stdrt', 
 'tfmri_nback_all_14', 'tfmri_nback_all_23', 'tfmri_nback_all_43', 'tfmri_nback_all_82', 'tfmri_nback_all_88', 'tfmri_nback_all_163', 
 'tfmri_nback_all_203', 'tfmri_ma_acvn_b_scs_aylh', 'tfmri_saisvcg_bscs_pdrh', 'tfmri_saasvcg_bscs_pulh', 'tfmri_saigvcg_bscs_pulh',
 'tfmri_sst_all_beh_crs_nt', 'tfmri_nback_all_216', 'tfmri_rec_all_beh_newpl_fa', 'tfmri_ma_alrvn_b_scs_aylh', 'tfmri_ma_alrvn_b_scs_ayrh',
 'tfmri_ma_asrvn_b_scs_ayrh', 'tfmri_ma_alvcr_b_scs_tplh', 'tfmri_ma_alvcr_b_scs_aylh', 'tfmri_ma_alvcr_b_scs_ayrh', 'tfmri_ma_aclvn_b_scs_ayrh',
 'tfmri_ma_acmvn_b_scs_ayrh', 'tfmri_ma_acgml_b_scs_ayrh', 'tfmri_ma_rpvnfb_b_cds_roagelh', 'tfmri_ma_rpvnfb_b_cds_clmdfrrh', 
 'tfmri_ma_lvnfb_b_cds_clmdfrrh', 'tfmri_ma_lvnfb_b_cds_roatcgerh', 'tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_lrwpfb_rate', 
 'tfmri_mid_all_beh_lrwnfb_nt', 'tfmri_mid_all_beh_slpfb_nt', 'tfmri_mid_all_beh_llpfb_nt', 'tfmri_mid_all_beh_llpfb_rate', 
 'tfmri_mid_all_beh_llnfb_nt', 'tfmri_mid_all_beh_hrwpfb_nt', 'tfmri_mid_all_beh_hrwnfb_rate', 'tfmri_mid_all_beh_hlpfb_nt', 
 'tfmri_mid_all_beh_hlnfb_nt', 'tfmri_mid_all_beh_hlnfb_rate', 'tfmri_sacgvf_bscs_crbwmlh', 'tfmri_sacgvf_bscs_pulh', 'tfmri_sacgvf_bscs_tprh', 
 'tfmri_sacgvf_bscs_purh', 'tfmri_saisvcg_bscs_crbwmlh', 'tfmri_saisvcg_bscs_cdelh', 'tfmri_saisvcg_bscs_purh', 'tfmri_saasvcg_bscs_crbwmlh',
 'tfmri_saasvcg_bscs_tprh', 'tfmri_sacsvis_bscs_crbwmrh', 'tfmri_saigvcg_bscs_crbwmlh', 'tfmri_saigvcg_bscs_crbwmrh', 'tfmri_saigvcg_bscs_tprh', 
 'tfmri_saigvcg_bscs_cderh', 'tfmri_saigvcg_bscs_purh', 'tfmri_saigvcg_bscs_pdrh', 'tfmri_saigvis_bscs_crbwmlh', 'tfmri_saigvis_bscs_crbcxlh', 
 'tfmri_saigvis_bscs_cdelh', 'tfmri_saigvis_bscs_crbcxrh', 'tfmri_saigvis_bscs_tprh', 'tfmri_saigvis_bscs_purh', 'tfmri_saigvis_bscs_pdrh', 
 'tfmri_sst_nbeh_nruns', 'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crlg_nt', 'tfmri_sst_all_beh_incrgo_rt', 'tfmri_sst_all_beh_incrgo_stdrt',
 'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_crs_rt', 'tfmri_nback_all_13', 'tfmri_nback_all_27', 'tfmri_nback_all_36', 'tfmri_nback_all_51',
 'tfmri_nback_all_53', 'tfmri_nback_all_73', 'tfmri_nback_all_81', 'tfmri_nback_all_83', 'tfmri_nback_all_87', 'tfmri_nback_all_112', 'tfmri_nback_all_141', 
 'tfmri_nback_all_148', 'tfmri_nback_all_154', 'tfmri_nback_all_171', 'tfmri_nback_all_172', 'tfmri_nback_all_177', 'tfmri_nback_all_178', 
 'tfmri_nback_all_184', 'tfmri_nback_all_186', 'tfmri_nback_all_223', 'tfmri_nback_all_231', 'tfmri_nback_all_244', 'tfmri_nback_all_245', 'tfmri_rec_all_beh_newnf_hr',
 'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_newpl_hr', 'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_fa', 'tfmri_rec_all_beh_posface_pr', 
 'tfmri_rec_all_beh_posface_br', 'tfmri_rec_all_beh_posf_dpr', 'tfmri_rec_all_beh_neutface_pr', 'tfmri_rec_all_beh_negface_br', 'tfmri_rec_all_beh_place_br',
 'tfmri_rec_all_beh_place_dp', 'gdt_scr_values_wins', 'sit_scr_expr_mfinalrat', 'sit_scr_values_count1', 'sit_scr_values_count2', 'sit_scr_values_countflips',
 'nihtbx_pattern_agecorrected', 'nihtbx_picture_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_pattern_fc', 'nihtbx_reading_fc', 'lmt_scr_perc_correct'])]} # SFFS_RF

# %%
# Gather results....
log_results = []
pred_results = []

def apply_model(classifier, name):
    if hasattr(classifier, 'predict_proba'):
        clf = RandomizedSearchCV(pipe, combined_grid, cv=10, verbose=True,
                            n_jobs=-1, scoring=metrics, refit='neg_log_loss', n_iter=10000)
    else:
        clf = RandomizedSearchCV(pipe, combined_grid, cv=10, verbose=True,
                             n_jobs=-1, scoring=metrics, refit='roc_auc', n_iter=10000)

    # Fit on data
    best_model = get_results(name, clf, X_train, y_train, X_test, y_test)

    #name_best_model = name + '_' + 'best_model.pkl'
    #dump(best_model, open(name_best_model, 'wb'))
    #name_clf = name + '_clf.pkl'
    #dump(clf, open(name_clf, 'wb'))
    #name_feature_eng = name + '_feat_eng.pkl'
    #dump(best_model.named_steps['feature_eng'], open(name_feature_eng, 'wb'))

    # Learning curved for final model
    estimator = best_model
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    plot_learning_curve(estimator, name, X_train, y_train, cv=cv, n_jobs=4)
    plt.show()

    model = estimator.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    sensitivity, specificity, acc, recall, precision, f1, cohenkappa = get_label_results(y_test, y_predict)
    predict_results = {'classifier': classifier, 'acc': acc, 'recall': recall, 'precision': precision, 'sensitivity': sensitivity, 'specificity': specificity, 'f1': f1, 'cohen kappa': cohenkappa}
    pred_results_copy = predict_results.copy()
    pred_results.append(pred_results_copy)

    if hasattr(estimator, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        model_logloss = log_loss(y_test, y_prob)
        model_rocauc = roc_auc_score(y_test, y_prob[:, 1])
        print('Logloss threshold %.3f' % logloss_threshold)
        print('Base logloss %.3f' % base_logloss)
        print('LogLoss %.3f' % model_logloss)
        print('Model ROC AUC %.3f' % model_rocauc)

        logloss_results = {'classifier': classifier, 'threshold': logloss_threshold, 'base': base_logloss,
                       'model': model_logloss, 'roc auc': model_rocauc}
        log_results_copy = logloss_results.copy()
        log_results.append(log_results_copy)
# %%
# Logistic Regression....
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("Warning not shown")

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    base_rocauc = roc_auc_score(y_test, y_prob[:, 1])
    print('Base LogLoss %.3f' % base_logloss)
    print('Base ROC AUC %.3f' % base_rocauc)

pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)), # initializes pipeline
                        ('clf' , LogisticRegression(random_state=42,
                                                    max_iter=2000))])

# Create param grid.
model_grid = {'clf__penalty' : ['l1', 'l2', 'elasticnet'],
            'clf__C' : np.logspace(-3, 3, 50),
            'clf__l1_ratio': np.linspace(0, 1, 20),
            'clf__solver' : ['saga']}

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'LogisticRegression')
# %%
# LinearDiscriminantAnalysis....

import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
print("Warning not shown")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
classifier = LinearDiscriminantAnalysis()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    print('Base LogLoss', np.round(log_loss(y_test, y_prob), 3))

pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)),
                         ('clf', LinearDiscriminantAnalysis())])

# Create param grid.
model_grid = {'clf__solver': ['svd', 'lsqr', 'eigen'],
               # 'clf__shrinkage': ['auto', None]
            }

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'LinearDA')

# Compute permutation feature importance
result = permutation_importance(classifier, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean,
    'Std Dev': result.importances_std
}).sort_values(by='Importance', ascending=False).head(25)
# Plot feature importances using viridis palette
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', xerr=importance_df['Std Dev'])
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importances for Linear Discriminant Analysis Model')
# Determine the desktop path
desktop = os.path.join(os.path.expanduser("/Users/madeleineseitz"), "Desktop")
filename = os.path.join(desktop, "feature_importances_lda.png")
# Save the plot to the desktop
plt.savefig(filename, bbox_inches='tight')
plt.show()
# %%
# QuadraticDiscriminantAnalysis....
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifier = QuadraticDiscriminantAnalysis()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    print('Base LogLoss', np.round(log_loss(y_test, y_prob), 3))

pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)),
                         ('clf', QuadraticDiscriminantAnalysis())])

# Create param grid.
model_grid = {'clf__reg_param': np.random.uniform(0, 0.95, size=100)
            }

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'QuadraticDA')

# Compute permutation feature importance
result = permutation_importance(classifier, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean,
    'Std Dev': result.importances_std
}).sort_values(by='Importance', ascending=False).head(25)
# Plot feature importances using viridis palette
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', xerr=importance_df['Std Dev'])
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importances for Quadratic Discriminant Analysis Model')
# Save the plot to the desktop
filename = os.path.join(desktop, "feature_importances_qda.png")
plt.savefig(filename, bbox_inches='tight')
plt.show()
# %%
# GaussianNB....
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    print('Base LogLoss', np.round(log_loss(y_test, y_prob), 3))

pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)),
                         ('clf', GaussianNB())])

# Create param grid.
model_grid = {'clf__var_smoothing': np.logspace(-10, 4, 50)
            }

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'GaussianNB')

# Compute permutation feature importance
result = permutation_importance(classifier, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean,
    'Std Dev': result.importances_std
}).sort_values(by='Importance', ascending=False).head(25)
# Plot feature importances using viridis palette
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', xerr=importance_df['Std Dev'])
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importances for Gaussian Naive Bayes Model')
# Save the plot to the desktop
filename = os.path.join(desktop, "feature_importances_gnb.png")
plt.savefig(filename, bbox_inches='tight')
plt.show()
# %%
# kNearestNeighbours....
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    print('Base LogLoss', np.round(log_loss(y_test, y_prob), 3))

pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)),
                         ('clf', KNeighborsClassifier())])

# Create param grid.
model_grid = {'clf__n_neighbors': np.round(np.linspace(3, 150, 50),0).astype(int),
            'clf__metric': ['minkowski', 'euclidean', 'manhattan']
            }

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'kNN')

# Compute permutation feature importance
result = permutation_importance(classifier, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean,
    'Std Dev': result.importances_std
}).sort_values(by='Importance', ascending=False).head(25)
# Plot feature importances using viridis palette
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', xerr=importance_df['Std Dev'])
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importances for K Nearest Neighbor Model')
# Save the plot to the desktop
filename = os.path.join(desktop, "feature_importances_knn.png")
plt.savefig(filename, bbox_inches='tight')
plt.show()
# %%
# DecisionTree Classifier....
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    print('Base LogLoss', np.round(log_loss(y_test, y_prob), 3))


pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)),
                         ('clf', DecisionTreeClassifier(random_state=42))])

# Create param grid.
model_grid = {'clf__splitter': ['best', 'random'],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_features': ['auto', 'sqrt', 'log2', 20, 10, 8, 5],
            'clf__max_depth': np.linspace(3, 30, 10).astype(int),
              'clf__max_leaf_nodes': np.linspace(50, 400, 50).astype(int),
              'clf__min_samples_split': np.random.randint(10, 30, 5)
            }

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'DecisionTree')

# Get feature importances
importances = classifier.feature_importances_
# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(25)
# Plot feature importances using viridis palette
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importances for Decision Tree Model')
# Save the plot to the desktop
filename = os.path.join(desktop, "feature_importances_decision_tree.png")
plt.savefig(filename, bbox_inches='tight')
plt.show()
# %%
# RandomForest Classifier....
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    print('Base LogLoss', np.round(log_loss(y_test, y_prob), 3))

pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)),
                         ('clf', RandomForestClassifier(random_state=42))])

# Create param grid.
model_grid = {'clf__n_estimators': [30, 50, 100, 200, 250],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_features': [12, 10, 8, 5],
            'clf__max_depth': np.linspace(3, 12, 4).astype(int),
              'clf__max_leaf_nodes': np.linspace(50, 200, 50).astype(int),
              'clf__min_samples_split': np.random.randint(10, 30, 5)
            }

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'RandomForest')

# Get feature importances
importances = classifier.feature_importances_
# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(25)
# Plot feature importances using viridis palette
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importances for Random Forest Model')
# Save the plot to the desktop
filename = os.path.join(desktop, "feature_importances_random_forest.png")
plt.savefig(filename, bbox_inches='tight')
plt.show()
# %%
# ExtraTrees Classifier....
from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    print('Base LogLoss', np.round(log_loss(y_test, y_prob), 3))

pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)),
                         ('clf', ExtraTreesClassifier(random_state=42))])

# Create param grid.
model_grid = {'clf__n_estimators': [30, 50, 100, 200, 250],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_features': [12, 10, 8, 5],
            'clf__max_depth': np.linspace(3, 12, 4).astype(int),
              'clf__max_leaf_nodes': np.linspace(50, 200, 50).astype(int),
              'clf__min_samples_split': np.random.randint(10, 30, 5)
            }

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'ExtraTrees')

# Get feature importances
importances = classifier.feature_importances_
# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(25)
# Plot feature importances using viridis palette
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importances for Extra Trees Model')
# Save the plot to the desktop
filename = os.path.join(desktop, "feature_importances_extra_trees.png")
plt.savefig(filename, bbox_inches='tight')
plt.show()
# %%
# XGBoost Classifier....
import xgboost as xgb
from xgboost import XGBClassifier
classifier = XGBClassifier()

# for comparison
base_model = classifier.fit(X_train, y_train)
if hasattr(classifier, 'predict_proba'):
    y_prob = base_model.predict_proba(X_test)
    base_logloss = log_loss(y_test, y_prob)
    print('Base LogLoss', np.round(log_loss(y_test, y_prob), 3))

pipe = Pipeline(steps = [('feature_eng', PCA(n_components=0.95)),
                         ('clf', XGBClassifier(random_state=42))])

# Create param grid.
model_grid = {'clf__booster': ['gbtree', 'gblinear'],
            'clf__learning_rate': np.random.uniform(0.001, 1, size=10),
            'clf__n_estimators': [50, 100, 200],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_features': [20, 10, 8, 5],
            'clf__max_depth': np.linspace(3, 30, 10).astype(int),
            'clf__max_leaf_nodes': np.linspace(50, 400, 50).astype(int),
            'clf__min_samples_split': np.random.randint(10, 30, 5)
            }

combined_grid = {**feature_grid, **model_grid}

apply_model(classifier, 'XGBoost')

# Get feature importances
importances = classifier.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(25)
# Plot feature importances using viridis palette
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importances for XGBoost Model')
# Save the plot to the desktop
filename = os.path.join(desktop, "feature_importances_xgboost.png")
plt.savefig(filename, bbox_inches='tight')
plt.show()
# %%
