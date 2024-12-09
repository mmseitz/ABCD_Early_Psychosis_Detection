#%%
#This script separates participants into high- and low- psychosisrisk groups
#High risk groups are defined as having endorsed 4+ PQ-BC questions at T2

import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np
# %%
####
#separate each df into separate timepoints
def timepoint_split(df, timepoint):
    split_df = df.loc[lambda df: df['visit'] == timepoint]
    split_df = split_df.reset_index()
    split_df = split_df.drop(columns= ['index'])
    return split_df

# %%
####
#Make dictionary of high and low risk participants (keys) and risk level based on # endorsed PQ-BC questions (value)
def make_risk_dic(df, risk_threshold):
    risk_dic = {}
    for row in df.index: 
        if df['PQBC_total_endorsed'][row] >= int(risk_threshold):
            risk_dic[df['subjectkey'][row]] = 'HR'
        else:
            risk_dic[df['subjectkey'][row]] = 'LR'
    return risk_dic

# %%
####
#find how many participants stayed in HR/LR groups between timepoints
def count_group_cont(first_dict, second_dict, timepoint1, timepoint2, threshold):
    df1_HR_count = 0
    df2_HR_count = 0
    df1_LR_count = 0
    df2_LR_count = 0
    HR_cont = []
    HR_drop = []
    LR_cont = []
    LR_drop = []
    for key in first_dict:
        if first_dict[key] == 'HR':
            df1_HR_count += 1
        else:
            df1_LR_count += 1
    for key in second_dict:
        if second_dict[key] == 'HR':
            df2_HR_count += 1
        else:
            df2_LR_count += 1
    print('Time point ' + timepoint1 + ' HR count')
    print(df1_HR_count)
    print('Time point ' + timepoint1 + ' LR count')
    print(df1_LR_count)
    print('Time point ' + timepoint2 + ' HR count')
    print(df2_HR_count)
    print('Time point ' + timepoint2 + ' LR count')
    print(df2_LR_count)

    for key in first_dict:
        if key in second_dict:
            if first_dict[key] == 'HR' and second_dict[key] == 'HR':
                HR_cont.append(key)
            if first_dict[key] == 'LR' and second_dict[key] == 'LR':
                LR_cont.append(key)
        else:
            if first_dict[key] == "HR":
                HR_drop.append(key)
            if first_dict[key] == "LR":
                LR_drop.append(key)
    print('risk threshold ' + threshold)
    print('HR cont ')
    print(len(HR_cont))
    print('HR drop ')
    print(len(HR_drop))
    print('LR cont ')
    print(len(LR_cont))
    print('LR drop ')
    print(len(LR_drop))

# %%
#Make dictionary of endorsement across categories
def calc_pop_endorsement(df, time_point, threshold):
    stats_dic = {}
    stats_dic['HR_count'] = 0
    stats_dic['HR_mean_PQBC_total'] = 0
    stats_dic['HR_bother_count'] = 0
    stats_dic['HR_pqbc_bother_avg'] = 0
    stats_dic['HR_intensity_sum_average'] = 0
    stats_dic['HR_mean_intensity_per_q_average'] = 0
    stats_dic['LR_count'] = 0
    stats_dic['LR_mean_PQBC_total'] = 0
    stats_dic['LR_bother_count'] = 0
    stats_dic['LR_pqbc_bother_avg'] = 0
    stats_dic['LR_intensity_sum_average'] = 0
    stats_dic['LR_mean_intensity_per_q_average'] = 0
    for row in df.index:
        if df['PQBC_total_endorsed'][row] >= int(threshold):
            stats_dic['HR_count'] += 1
            stats_dic['HR_mean_PQBC_total'] +=int(df['PQBC_total_endorsed'][row])
            if df['PQBC_total_bother'][row] >= 1:
                stats_dic['HR_bother_count'] += 1
                stats_dic['HR_pqbc_bother_avg'] += int(df['PQBC_total_bother'][row])
            if df['PQBC_total_bother_intensity'][row] >= 0:
                stats_dic['HR_intensity_sum_average'] += int(df['PQBC_total_bother_intensity'][row])
                stats_dic['HR_mean_intensity_per_q_average'] += float(df['mean_intensity_per_bother_Q'][row])
        else:
            stats_dic['LR_count'] += 1
            stats_dic['LR_mean_PQBC_total'] +=int(df['PQBC_total_endorsed'][row])
            if df['PQBC_total_bother'][row] >= 1:
                stats_dic['LR_bother_count'] += 1
                stats_dic['LR_pqbc_bother_avg'] += int(df['PQBC_total_bother'][row])
            if df['PQBC_total_bother_intensity'][row] >= 0:
                stats_dic['LR_intensity_sum_average'] += int(df['PQBC_total_bother_intensity'][row])
                stats_dic['LR_mean_intensity_per_q_average'] += float(df['mean_intensity_per_bother_Q'][row])
    stats_dic['HR_mean_PQBC_total'] = stats_dic['HR_mean_PQBC_total']/stats_dic['HR_count']
    stats_dic['HR_pqbc_bother_avg'] = stats_dic['HR_pqbc_bother_avg']/stats_dic['HR_bother_count']
    stats_dic['HR_intensity_sum_average'] = stats_dic['HR_intensity_sum_average']/stats_dic['HR_bother_count']
    stats_dic['HR_mean_intensity_per_q_average'] = stats_dic['HR_mean_intensity_per_q_average']/stats_dic['HR_bother_count']
    stats_dic['LR_mean_PQBC_total'] = stats_dic['LR_mean_PQBC_total']/stats_dic['LR_count']
    stats_dic['LR_pqbc_bother_avg'] = stats_dic['LR_pqbc_bother_avg']/stats_dic['LR_bother_count']
    stats_dic['LR_intensity_sum_average'] = stats_dic['LR_intensity_sum_average']/stats_dic['LR_bother_count']
    stats_dic['LR_mean_intensity_per_q_average'] = stats_dic['LR_mean_intensity_per_q_average']/stats_dic['LR_bother_count']
    print('Frequency of population endorsement for timepoint ' + time_point + ' and bother threshold ' + threshold)
    return(stats_dic)

# %%
#read in csvs containing pq-bc totals
pps_raw = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_y_pps.csv')
# %%
#Sum total number of PQBC questions endorsed by each participant, sum total number of PQBC questions participants endorsed being
#bothered by, sum total distress across all PQBC questions, calculate mean distress score per endorsed PQBC question
pps_all = pps_raw
#pps_all = pps_all.drop([0])
pps_all = pps_all.drop(columns= ['pps_y_ss_number', 'pps_y_ss_number_nm', 'pps_y_ss_number_nt', 'pps_y_ss_bother_sum', 'pps_y_ss_bother_sum_nm', 
                                 'pps_y_ss_bother_sum_nt', 'pps_y_ss_bother_n_1', 'pps_y_ss_bother_n_1_nm', 'pps_y_ss_bother_n_1_nt', 
                                 'pps_y_ss_severity_score', 'pps_y_ss_severity_score_nm', 'pps_y_ss_severity_score_nt', 'pps_ss_mean_severity'])
#replace NaN values with to aid operations
pps_all = pps_all.fillna(0)
# %%
#identify columns pertaining to each question part
endorsed = ['prodromal_1_y', 'prodromal_2_y', 'prodromal_3_y', 'prodromal_4_y', 'prodromal_5_y', 'prodromal_6_y', 'prodromal_7_y',
            'prodromal_8_y', 'prodromal_9_y', 'prodromal_10_y', 'prodromal_11_y', 'prodromal_12_y', 'prodromal_13_y', 'prodromal_14_y',
              'prodromal_15_y', 'prodromal_16_y', 'prodromal_17_y', 'prodromal_18_y', 'prodromal_19_y', 'prodromal_20_y', 'prodromal_21_y']
bother = ['pps_1_bother_yn', 'pps_2_bother_yn', 'pps_3_bother_yn', 'pps_4_bother_yn', 'pps_5_bother_yn', 'pps_6_bother_yn', 'pps_7_bother_yn', 
          'pps_8_bother_yn', 'pps_9_bother_yn', 'pps_10_bother_yn', 'pps_11_bother_yn', 'pps_12_bother_yn', 'pps_13_bother_yn', 'pps_14_bother_yn', 
          'pps_15_bother_yn', 'pps_16_bother_yn', 'pps_17_bother_yn', 'pps_18_bother_yn', 'pps_19_bother_yn', 'pps_20_bother_yn', 'pps_21_bother_yn']
intensity = ['prodromal_1b_y', 'prodromal_2b_y', 'prodromal_3b_y', 'prodromal_4b_y', 'prodromal_5b_y', 'prodromal_6b_y', 'prodromal_7b_y',
            'prodromal_8b_y', 'prodromal_9b_y', 'prodromal_10b_y', 'prodromal_11b_y', 'prodromal_12b_y', 'prodromal_13b_y', 'prodromal_14b_y',
              'prodromal_15b_y', 'prodromal_16b_y', 'prodromal_17b_y', 'prodromal_18b_y', 'prodromal_19b_y', 'prodromal_20b_y', 'prodromal_21b_y']
# %%
#convert str to float 
pps_endorsed = pps_all[endorsed].apply(pd.to_numeric)
pps_bother = pps_all[bother].apply(pd.to_numeric)
pps_intensity = pps_all[intensity].apply(pd.to_numeric)
# %%
#create new columns containing totals
pps_sums = pps_all
pps_sums['PQBC_total_endorsed'] = pps_endorsed.sum(axis=1)
pps_sums['PQBC_total_bother'] = pps_bother.sum(axis=1)
pps_sums['PQBC_total_bother_intensity'] = pps_intensity.sum(axis=1)
pps_sums['mean_intensity_per_bother_Q'] = pps_all['PQBC_total_bother_intensity']/pps_all['PQBC_total_bother']
# %%
#clean and save df containing raw information and sum columns
#pps_sums = pps_sums.drop([0])
pps_sums = pps_sums.fillna(0)
pps_sums.to_csv('/Users/madeleineseitz/Desktop/pps_5_sums.csv')
#shorten df to contain only sums and subject identification info
# %%
pps_short = pps_sums[['src_subject_id', 'eventname', 'PQBC_total_endorsed', 'PQBC_total_bother', 'PQBC_total_bother_intensity', 'mean_intensity_per_bother_Q']]
pps_short = pps_short.rename(columns= {'eventname': 'visit', 'src_subject_id': 'subjectkey'})
# %%
#rename visit timepoints to [0, 1, 2, 3]
pps_short.loc[pps_short['visit'] == 'baseline_year_1_arm_1', ['visit']] = 0
pps_short.loc[pps_short['visit'] == '1_year_follow_up_y_arm_1', ['visit']] = 1
pps_short.loc[pps_short['visit'] == '2_year_follow_up_y_arm_1', ['visit']] = 2
pps_short.loc[pps_short['visit'] == '3_year_follow_up_y_arm_1', ['visit']] = 3
pps_short.loc[pps_short['visit'] == '4_year_follow_up_y_arm_1', ['visit']] = 4
# %%
#separate into sub-dfs by timepoint
pps_bl = timepoint_split(pps_short, 0)
pps_1 = timepoint_split(pps_short, 1)
pps_2 = timepoint_split(pps_short, 2)
pps_3 = timepoint_split(pps_short, 3)
pps_4 = timepoint_split(pps_short, 4)
# %%
#make dictionaries of high-risk participants
risk_bl = make_risk_dic(pps_bl, 4)
risk_1 = make_risk_dic(pps_1, 4)
risk_2 = make_risk_dic(pps_2, 4)
risk_3 = make_risk_dic(pps_3, 4)
risk_4 = make_risk_dic(pps_4, 4)
# %%
#calculate how many high risk participants continued from group to group across timepoints
#count_group_cont(risk_bl, risk_1, 'baseline', '1', '4')
#count_group_cont(risk_1, risk_2, '1', '2', '4')
#count_group_cont(risk_2, risk_3, '2', '3', '4')
count_group_cont(risk_3, risk_4, '3', '4', '4')
# %%
#add high risk "group" column to pps_short df and pps_2 df and pps_4 df
#threshold defined at 2
pps_short_t2 = pps_short
HR_list = []
for key in risk_2:
    if risk_2[key] == 'HR':
        HR_list.append(key)
        len(HR_list)
pps_short_t2['group'] = 0
for row in pps_short_t2.index:
    if pps_short_t2['subjectkey'][row] in HR_list:
        pps_short_t2['group'][row] = 1
pps_2['group'] = 0
for row in pps_2.index:
    if pps_2['subjectkey'][row] in HR_list:
        pps_2['group'][row] = 1
pps_2['group'].sum()
#pps_short_t2 = pps_short_t2.drop([0])
# %%
pps_short_t2.to_csv('/Users/madeleineseitz/Desktop/PQBC_T2_thresh4_ABCD_5.csv')
pps_2.to_csv('/Users/madeleineseitz/Desktop/PQBC_T2_grouped_ABCD_5.csv')
# %%
#threshold defined at 4
pps_short_t4 = pps_short
HR_list = []
for key in risk_4:
    if risk_4[key] == 'HR':
        HR_list.append(key)
        len(HR_list)
pps_short_t4['group'] = 0
for row in pps_short_t4.index:
    if pps_short_t4['subjectkey'][row] in HR_list:
        pps_short_t4['group'][row] = 1
pps_4['group'] = 0
for row in pps_4.index:
    if pps_4['subjectkey'][row] in HR_list:
        pps_4['group'][row] = 1
pps_4['group'].sum()
#pps_short_t4 = pps_short_t4.drop([0])
# %%
#save df to file
pps_short_t4.to_csv('/Users/madeleineseitz/Desktop/PQBC_T4_thresh4_ABCD_5.csv')
pps_4.to_csv('/Users/madeleineseitz/Desktop/PQBC_T4_grouped_ABCD_5.csv')
# %%
pps_2_HR = pps_2.loc[lambda pps_2: pps_2['group'] == 1]
pps_2_LR = pps_2.loc[lambda pps_2: pps_2['group'] == 0]
# %%
calc_pop_endorsement(pps_bl, 'baseline', '4')
# %%
calc_pop_endorsement(pps_1, '1', '4')
# %%
calc_pop_endorsement(pps_2, '2', '4')
# %%
calc_pop_endorsement(pps_3, '3', '4')
# %%
calc_pop_endorsement(pps_4, '4', '4')
# %%
