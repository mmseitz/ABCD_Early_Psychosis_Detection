""" This script creates a brief demographic breakdown of matched T2 and T4 datasets"""
# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy.stats import chi2_contingency

# %%
#Import csvs (preprocessed T2 and T4 datasets--matched for ethnicity, sex, and age, no NaN values
# all features)

abcd_t2 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/uniform_csvs/T2/abcd5_T2_uniform_abcd_nonan.csv')
abcd_t2 = abcd_t2.drop(columns= ['Unnamed: 0'])
abcd_t4 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/uniform_csvs/T4/abcd5_T4_uniform_abcd_nonan.csv')
abcd_t4 = abcd_t4.drop(columns= ['Unnamed: 0'])
# %%
#Demographics of T2 data
t2_hr = abcd_t2.loc[abcd_t2['group'] == 1] #HR count 565
t2_lr = abcd_t2.loc[abcd_t2['group'] == 0] #LR count 612

# %%
#calculate mean age
t2_hr['interview_age'].mean()
#t2_lr['interview_age'].mean()

# %%
#Sex Count
hr_male_count = t2_hr['sex'].isin([1]).sum()
hr_female_count = t2_hr['sex'].isin([2]).sum()
lr_male_count = t2_lr['sex'].isin([1]).sum()
lr_female_count = t2_lr['sex'].isin([2]).sum()

print('High Risk male count:', hr_male_count) #287 participants
print('High Risk female count:', hr_female_count) #278 participants
print('Low Risk male count:', lr_male_count) #316 participants
print('Low Risk female count:', lr_female_count) #296 participants

# %%
#Income Count
hr_low_count = t2_hr['income'].isin([1]).sum()
hr_med_count = t2_hr['income'].isin([2]).sum()
hr_high_count = t2_hr['income'].isin([3]).sum()
lr_low_count = t2_lr['income'].isin([1]).sum()
lr_med_count = t2_lr['income'].isin([2]).sum()
lr_high_count = t2_lr['income'].isin([3]).sum()

print('High Risk low income count:', hr_low_count) #147 participants
print('High Risk med income count:', hr_med_count) #260 participants
print('High Risk high income count:', hr_high_count) #158 participants
print('Low Risk low income count:', lr_low_count) #122 participants
print('Low Risk med income count:', lr_med_count) #214 participants
print('Low Risk high income count:', lr_high_count) #276 participants

# %%
#Parent Education Count
hr_low_count = t2_hr['p_edu'].isin([1]).sum()
hr_med_count = t2_hr['p_edu'].isin([2]).sum()
hr_high_count = t2_hr['p_edu'].isin([3]).sum()
lr_low_count = t2_lr['p_edu'].isin([1]).sum()
lr_med_count = t2_lr['p_edu'].isin([2]).sum()
lr_high_count = t2_lr['p_edu'].isin([3]).sum()

print('High high school count:', hr_low_count) #81 participants
print('High college count:', hr_med_count) #350 participants
print('High post grad count:', hr_high_count) #134 participants
print('Low high school count:', lr_low_count) #83 participants
print('Low Risk college count:', lr_med_count) #340 participants
print('Low Risk post grad count:', lr_high_count) #189 participants
# %%
dems_dict = {}
contingency_table1 = pd.crosstab(abcd_t2['group'], abcd_t2['sex'])
dems_dict['sex_chi'] = chi2_contingency(contingency_table1) # p = 0.819 (groups matched with R Matchit)
contingency_table2 = pd.crosstab(abcd_t2['group'], abcd_t2['income'])
dems_dict['income_chi'] = chi2_contingency(contingency_table2) # p = 8.9967E-09 
contingency_table3 = pd.crosstab(abcd_t2['group'], abcd_t2['p_edu'])
dems_dict['p_edu_chi'] = chi2_contingency(contingency_table3) # p = 0.0216
contingency_table4 = pd.crosstab(abcd_t2['group'], abcd_t2['race'])
dems_dict['ethnicity_chi'] = chi2_contingency(contingency_table4) # p = 0.7748 (groups matched with R Matchit)
print(dems_dict)
# %%
#ACE frequency
ale_y_dict = {}
ale_y_dict['ALE_total_welch'] = stats.ttest_ind(t2_hr['ple_y_ss_total_number'], t2_lr['ple_y_ss_total_number']) # p = 7.5836E-59
ale_y_dict['ALE_total_good'] = stats.ttest_ind(t2_hr['ple_y_ss_total_good'], t2_lr['ple_y_ss_total_good']) # p = 5.4335E-07
ale_y_dict['ALE_total_bad'] = stats.ttest_ind(t2_hr['ple_y_ss_total_bad'], t2_lr['ple_y_ss_total_bad']) # p = 1.0548E-52
ale_y_dict['ALE_affected_good'] = stats.ttest_ind(t2_hr['ple_y_ss_affected_good_sum'], t2_lr['ple_y_ss_affected_good_sum']) #p = 0.0012
ale_y_dict['ALE_affected_bad'] = stats.ttest_ind(t2_hr['ple_y_ss_affected_bad_sum'], t2_lr['ple_y_ss_affected_bad_sum']) # p = 8.3208E-50
print(ale_y_dict)

# %%
#ACE means
total_ale_mean = t2_hr['ple_y_ss_total_number'].mean()
total_ale_bad_mean = t2_hr['ple_y_ss_total_bad'].mean()
total_ale_affected_bad = t2_hr['ple_y_ss_affected_bad_sum'].mean()
total_ale_mean_lr = t2_lr['ple_y_ss_total_number'].mean()
total_ale_bad_mean_lr = t2_lr['ple_y_ss_total_bad'].mean()
total_ale_affected_bad_lr = t2_lr['ple_y_ss_affected_bad_sum'].mean()

print('hr total ACE mean', total_ale_mean) # 7.2832
print('hr total ACE bad mean', total_ale_bad_mean) # 3.7947
print('hr total ACE affected bad mean', total_ale_affected_bad) # 8.1965
print('total ACE mean LR', total_ale_mean_lr) # 4.219
print('total ACE bad mean LR', total_ale_bad_mean_lr) # 1.7206
print('total ACE affected bad mean LR', total_ale_affected_bad_lr) # 3.3268
# %%
#Peer Experiences Questionnaire (Youth) statistical analyses between HR/LR groups
#add HR/LR groupings to peq df
peq_cols = ['peq_ss_relational_victim' , 'peq_ss_reputation_aggs', 'peq_ss_reputation_victim', 'peq_ss_overt_aggression', 'peq_ss_overt_victim', 'peq_ss_relational_aggs']
peq_dict = {}

for i in peq_cols:
    peq_dict[i] = stats.ttest_ind(t2_hr[i], t2_lr[i], equal_var = False)
print(peq_dict)

#relational victim p = 9.1276E-37
#reputational aggression p = 8.3199
#reputation victim p = 1.2258E-34
#overt aggression p = 1.62599E-16
#overt victim p = 1.1689E-29
#relational aggression p = 1.2351E-14
# %%
""" Same measures on T4 data"""
t4_hr = abcd_t4.loc[abcd_t4['group'] == 1] #HR count 109
t4_lr = abcd_t4.loc[abcd_t4['group'] == 0] #LR count 100

# %%
#calculate mean age
t4_hr['interview_age'].mean()
t4_lr['interview_age'].mean()

# %%
#Sex Count
hr_male_count = t4_hr['sex'].isin([1]).sum()
hr_female_count = t4_hr['sex'].isin([2]).sum()
lr_male_count = t4_lr['sex'].isin([1]).sum()
lr_female_count = t4_lr['sex'].isin([2]).sum()

print('High Risk male count:', hr_male_count) #39 participants
print('High Risk female count:', hr_female_count) #70 participants
print('Low Risk male count:', lr_male_count) #41 participants
print('Low Risk female count:', lr_female_count) #59 participants

# %%
#Income Count
hr_low_count = t4_hr['income'].isin([1]).sum()
hr_med_count = t4_hr['income'].isin([2]).sum()
hr_high_count = t4_hr['income'].isin([3]).sum()
lr_low_count = t4_lr['income'].isin([1]).sum()
lr_med_count = t4_lr['income'].isin([2]).sum()
lr_high_count = t4_lr['income'].isin([3]).sum()

print('High Risk low income count:', hr_low_count) #24 participants
print('High Risk med income count:', hr_med_count) #52 participants
print('High Risk high income count:', hr_high_count) #33 participants
print('Low Risk low income count:', lr_low_count) #23 participants
print('Low Risk med income count:', lr_med_count) #29 participants
print('Low Risk high income count:', lr_high_count) #48 participants

# %%
#Parent Education Count
hr_low_count = t4_hr['p_edu'].isin([1]).sum()
hr_med_count = t4_hr['p_edu'].isin([2]).sum()
hr_high_count = t4_hr['p_edu'].isin([3]).sum()
lr_low_count = t4_lr['p_edu'].isin([1]).sum()
lr_med_count = t4_lr['p_edu'].isin([2]).sum()
lr_high_count = t4_lr['p_edu'].isin([3]).sum()

print('High high school count:', hr_low_count) #17 participants
print('High college count:', hr_med_count) #67 participants
print('High post grad count:', hr_high_count) #25 participants
print('Low high school count:', lr_low_count) #10 participants
print('Low Risk college count:', lr_med_count) #57 participants
print('Low Risk post grad count:', lr_high_count) #33 participants
# %%
dems_dict = {}
contingency_table1 = pd.crosstab(abcd_t4['group'], abcd_t4['sex'])
dems_dict['sex_chi'] = chi2_contingency(contingency_table1) # p = 0.5266 (groups matched with R Matchit)
contingency_table2 = pd.crosstab(abcd_t4['group'], abcd_t4['income'])
dems_dict['income_chi'] = chi2_contingency(contingency_table2) # p = 0.0113 
contingency_table3 = pd.crosstab(abcd_t4['group'], abcd_t4['p_edu'])
dems_dict['p_edu_chi'] = chi2_contingency(contingency_table3) # p = 0.1879
contingency_table4 = pd.crosstab(abcd_t4['group'], abcd_t4['race'])
dems_dict['ethnicity_chi'] = chi2_contingency(contingency_table4) # p = 0.3416 (groups matched with R Matchit)
print(dems_dict)
# %%
#ACE frequency
ale_y_dict = {}
ale_y_dict['ALE_total_welch'] = stats.ttest_ind(t4_hr['ple_y_ss_total_number'], t4_lr['ple_y_ss_total_number']) # p = 7.9223E-07
ale_y_dict['ALE_total_good'] = stats.ttest_ind(t4_hr['ple_y_ss_total_good'], t4_lr['ple_y_ss_total_good']) # p = 0.266
ale_y_dict['ALE_total_bad'] = stats.ttest_ind(t4_hr['ple_y_ss_total_bad'], t4_lr['ple_y_ss_total_bad']) # p = 2.3267E-07
ale_y_dict['ALE_affected_good'] = stats.ttest_ind(t4_hr['ple_y_ss_affected_good_sum'], t4_lr['ple_y_ss_affected_good_sum']) #p = 0.632
ale_y_dict['ALE_affected_bad'] = stats.ttest_ind(t4_hr['ple_y_ss_affected_bad_sum'], t4_lr['ple_y_ss_affected_bad_sum']) # p = 9.9805E-08
print(ale_y_dict)

# %%
#ACE means
total_ale_mean = t4_hr['ple_y_ss_total_number'].mean()
total_ale_bad_mean = t4_hr['ple_y_ss_total_bad'].mean()
total_ale_affected_bad = t4_hr['ple_y_ss_affected_bad_sum'].mean()
total_ale_mean_lr = t4_lr['ple_y_ss_total_number'].mean()
total_ale_bad_mean_lr = t4_lr['ple_y_ss_total_bad'].mean()
total_ale_affected_bad_lr = t4_lr['ple_y_ss_affected_bad_sum'].mean()

print('hr total ACE mean', total_ale_mean) # 5.9817
print('hr total ACE bad mean', total_ale_bad_mean) # 3.0183
print('hr total ACE affected bad mean', total_ale_affected_bad) # 6.422
print('total ACE mean LR', total_ale_mean_lr) # 3.7
print('total ACE bad mean LR', total_ale_bad_mean_lr) # 1.52
print('total ACE affected bad mean LR', total_ale_affected_bad_lr) # 2.74
# %%
#Peer Experiences Questionnaire (Youth) statistical analyses between HR/LR groups
#add HR/LR groupings to peq df
peq_cols = ['peq_ss_relational_victim' , 'peq_ss_reputation_aggs', 'peq_ss_reputation_victim', 'peq_ss_overt_aggression', 'peq_ss_overt_victim', 'peq_ss_relational_aggs']
peq_dict = {}

for i in peq_cols:
    peq_dict[i] = stats.ttest_ind(t4_hr[i], t4_lr[i], equal_var = False)
print(peq_dict)

#relational victim p = 2.8814E-06
#reputational aggression p = 0.5376
#reputation victim p = 4.3074E-08
#overt aggression p = 0.0001
#overt victim p = 1.9953E-05
#relational aggression p = 0.1158

# %%
