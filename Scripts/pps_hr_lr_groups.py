#%%
#This script separates participants into high or low psychosis risk groups based on how many PQ-BC questions they endorsed bothered them
#This script also generates some simple statistics representing the distribution of endorsement across timepoints

import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np
# %%
#read in csvs containing pq-bc totals

screen_totals = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/screener_totals.csv')
screen_totals = screen_totals.drop(columns= ['Unnamed: 0'])

pqbc_totals = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/pqbc_totals.csv')
pqbc_totals = pqbc_totals.drop(columns= ['Unnamed: 0'])
# %%
#separate each df into separate timepoints
def timepoint_split(df, timepoint):
    split_df = df.loc[lambda df: df['visit'] == timepoint]
    split_df = split_df.reset_index()
    split_df = split_df.drop(columns= ['index'])
    return split_df

# %%
#calculate instances of endorsed questions in pqbc and screener by timepoint and group threshold
#enter operator "greater_than" or "equal_to"

def calc_pop_endorsement(df, time_point, threshold, operator):
    stats_dic = {}
    stats_dic['bother_thresh_count'] = 0
    stats_dic['pqbc_bother_total_avg'] = 0
    stats_dic['pqbc_yes_count'] = 0
    stats_dic['pqbc_y_total_avg'] = 0
    stats_dic['screen_yes_count'] = 0
    stats_dic['screen_y_total_avg_pos'] = 0
    stats_dic['screen_bother_counter'] = 0
    stats_dic['screen_bother_total_avg_pos'] = 0
    if operator == 'greater_than':
        for row in df.index:
            if df['pqbc_bother_totals'][row] >= int(threshold):
                stats_dic['bother_thresh_count'] += 1
                stats_dic['pqbc_bother_total_avg'] += int(df['pqbc_bother_totals'][row])
                if df['pqbc_endorsed_totals'][row] >= 1:
                    stats_dic['pqbc_yes_count'] += 1
                    stats_dic['pqbc_y_total_avg'] += int(df['pqbc_endorsed_totals'][row])
                if df['screen_endorsed_totals'][row] >= 1:
                    stats_dic['screen_yes_count'] += 1
                    stats_dic['screen_y_total_avg_pos'] += int(df['screen_endorsed_totals'][row])
                if df['screen_bother_totals'][row] >= 1:
                    stats_dic['screen_bother_counter'] += 1
                    stats_dic['screen_bother_total_avg_pos'] += int(df['screen_bother_totals'][row])
    else:
        for row in df.index:
            if df['pqbc_bother_totals'][row] == int(threshold):
                stats_dic['bother_thresh_count'] += 1
                stats_dic['pqbc_bother_total_avg'] += int(df['pqbc_bother_totals'][row])
                if df['pqbc_endorsed_totals'][row] >= 1:
                    stats_dic['pqbc_yes_count'] += 1
                    stats_dic['pqbc_y_total_avg'] += int(df['pqbc_endorsed_totals'][row])
                if df['screen_endorsed_totals'][row] >= 1:
                    stats_dic['screen_yes_count'] += 1
                    stats_dic['screen_y_total_avg_pos'] += int(df['screen_endorsed_totals'][row])
                if df['screen_bother_totals'][row] >= 1:
                    stats_dic['screen_bother_counter'] += 1
                    stats_dic['screen_bother_total_avg_pos'] += int(df['screen_bother_totals'][row])
    stats_dic['pqbc_bother_total_avg'] = stats_dic['pqbc_bother_total_avg']/stats_dic['bother_thresh_count']
    stats_dic['pqbc_y_total_avg'] = stats_dic['pqbc_y_total_avg']/stats_dic['pqbc_yes_count']
    stats_dic['screen_y_total_avg_pop'] = stats_dic['screen_y_total_avg_pos']/stats_dic['bother_thresh_count']
    stats_dic['screen_bother_total_avg_pop'] = stats_dic['screen_bother_total_avg_pos']/stats_dic['bother_thresh_count']
    stats_dic['screen_y_total_avg_pos'] = stats_dic['screen_y_total_avg_pos']/stats_dic['screen_yes_count']
    stats_dic['screen_bother_total_avg_pos'] = stats_dic['screen_bother_total_avg_pos']/stats_dic['screen_bother_counter']
    del stats_dic['pqbc_yes_count']
    print('Frequency of population endorsement for timepoint ' + time_point + ' and bother threshold ' + threshold)
    return(stats_dic)

# %%
#Make dictionary of high and low risk participants (keys) and risk level based on PQ-BC bother threshold (value)
def make_risk_dic(df, risk_threshold):
    risk_dic = {}
    for row in df.index: 
        if df['pqbc_bother_totals'][row] >= int(risk_threshold):
            risk_dic[df['subjectkey'][row]] = 'HR'
        else:
            risk_dic[df['subjectkey'][row]] = 'LR'
    return risk_dic

# %%
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
#split screen_totals df by timepoint 
screen_bl = timepoint_split(screen_totals, 0)
screen_1 = timepoint_split(screen_totals, 1)
screen_2 = timepoint_split(screen_totals, 2)
screen_3 = timepoint_split(screen_totals, 3)
# %%
#split pqbc_totals df by timepoint
pqbc_bl = timepoint_split(pqbc_totals, 0)
pqbc_1 = timepoint_split(pqbc_totals, 1)
pqbc_2 = timepoint_split(pqbc_totals, 2)
pqbc_3 = timepoint_split(pqbc_totals, 3)

# %%
#make dfs with pqbc and screener totals
screen_3_short = screen_3[['subjectkey', 'visit', 'screen_endorsed_totals', 'screen_bother_totals']]
tp_3_totals = pqbc_3.merge(screen_3_short, on = ['subjectkey', 'visit'])

screen_2_short = screen_2[['subjectkey', 'visit', 'screen_endorsed_totals', 'screen_bother_totals']]
tp_2_totals = pqbc_2.merge(screen_2_short, on = ['subjectkey', 'visit'])

screen_1_short = screen_1[['subjectkey', 'visit', 'screen_endorsed_totals', 'screen_bother_totals']]
tp_1_totals = pqbc_1.merge(screen_1_short, on = ['subjectkey', 'visit'])

screen_bl_short = screen_bl[['subjectkey', 'visit', 'screen_endorsed_totals', 'screen_bother_totals']]
tp_bl_totals = pqbc_bl.merge(screen_bl_short, on = ['subjectkey', 'visit'])
# %%
#Stats recorded in PQ-BC_stats.xls
#Calculate df stats for baseline

#calc_pop_endorsement(tp_bl_totals, 'bl', '1', 'greater_than')
#calc_pop_endorsement(tp_bl_totals, 'bl', '1', 'equals_to')
#calc_pop_endorsement(tp_bl_totals, 'bl', '2', 'greater_than')
#calc_pop_endorsement(tp_bl_totals, 'bl', '2', 'equals_to')
#calc_pop_endorsement(tp_bl_totals, 'bl', '3', 'greater_than')
#calc_pop_endorsement(tp_bl_totals, 'bl', '3', 'equals_to')
#calc_pop_endorsement(tp_bl_totals, 'bl', '4', 'greater_than')
#calc_pop_endorsement(tp_bl_totals, 'bl', '4', 'equals_to')
#calc_pop_endorsement(tp_bl_totals, 'bl', '5', 'greater_than')
#calc_pop_endorsement(tp_bl_totals, 'bl', '5', 'equals_to')
tp_bl_totals.shape
# %%
#Calculate df stats for follow-up 1

#calc_pop_endorsement(tp_1_totals, '1', '1', 'greater_than')
#calc_pop_endorsement(tp_1_totals, '1', '1', 'equals_to')
#calc_pop_endorsement(tp_1_totals, '1', '2', 'greater_than')
#calc_pop_endorsement(tp_1_totals, '1', '2', 'equals_to')
#calc_pop_endorsement(tp_1_totals, '1', '3', 'greater_than')
#calc_pop_endorsement(tp_1_totals, '1', '3', 'equals_to')
#calc_pop_endorsement(tp_1_totals, '1', '4', 'greater_than')
#calc_pop_endorsement(tp_1_totals, '1', '4', 'equals_to')
#calc_pop_endorsement(tp_1_totals, '1', '5', 'greater_than')
#calc_pop_endorsement(tp_1_totals, '1', '5', 'equals_to')
tp_1_totals.shape

# %%
#Calculate df stats for follow-up 2

#calc_pop_endorsement(tp_2_totals, '2', '1', 'greater_than')
#calc_pop_endorsement(tp_2_totals, '2', '1', 'equals_to')
#calc_pop_endorsement(tp_2_totals, '2', '2', 'greater_than')
#calc_pop_endorsement(tp_2_totals, '2', '2', 'equals_to')
#calc_pop_endorsement(tp_2_totals, '2', '3', 'greater_than')
#calc_pop_endorsement(tp_2_totals, '2', '3', 'equals_to')
#calc_pop_endorsement(tp_2_totals, '2', '4', 'greater_than')
#calc_pop_endorsement(tp_2_totals, '2', '4', 'equals_to')
#calc_pop_endorsement(tp_2_totals, '2', '5', 'greater_than')
#calc_pop_endorsement(tp_2_totals, '2', '5', 'equals_to')
tp_2_totals.shape

# %%
#Calculate df stats for follow-up 3

#calc_pop_endorsement(tp_3_totals, '3', '1', 'greater_than')
#calc_pop_endorsement(tp_3_totals, '3', '1', 'equals_to')
#calc_pop_endorsement(tp_3_totals, '3', '2', 'greater_than')
#calc_pop_endorsement(tp_3_totals, '3', '2', 'equals_to')
#calc_pop_endorsement(tp_3_totals, '3', '3', 'greater_than')
#calc_pop_endorsement(tp_3_totals, '3', '3', 'equals_to')
#calc_pop_endorsement(tp_3_totals, '3', '4', 'greater_than')
#calc_pop_endorsement(tp_3_totals, '3', '4', 'equals_to')
#calc_pop_endorsement(tp_3_totals, '3', '5', 'greater_than')
#calc_pop_endorsement(tp_3_totals, '3', '5', 'equals_to')
tp_3_totals.shape

# %%
#make dictionaries with threshold set at bothered by 2+ pqbc questions
tbl_thresh_2_dic = make_risk_dic(tp_bl_totals, '2')
t1_thresh_2_dic = make_risk_dic(tp_1_totals, '2')
t2_thresh_2_dic = make_risk_dic(tp_2_totals, '2')
t3_thresh_2_dic = make_risk_dic(tp_3_totals, '2')

# %%
#make dictionaries with threshold set at bothered by 3+ pqbc questions
tbl_thresh_3_dic = make_risk_dic(tp_bl_totals, '3')
t1_thresh_3_dic = make_risk_dic(tp_1_totals, '3')
t2_thresh_3_dic = make_risk_dic(tp_2_totals, '3')
t3_thresh_3_dic = make_risk_dic(tp_3_totals, '3')

# %%
#make dictionaries with threshold set at bothered by 4+ pqbc questions
tbl_thresh_4_dic = make_risk_dic(tp_bl_totals, '4')
t1_thresh_4_dic = make_risk_dic(tp_1_totals, '4')
t2_thresh_4_dic = make_risk_dic(tp_2_totals, '4')
t3_thresh_4_dic = make_risk_dic(tp_3_totals, '4')

# %%
#Stats recorded in PQ-BC_stats.xls
#test group continuity between baseline and follow-up 1
#count_group_cont(tbl_thresh_2_dic, t1_thresh_2_dic, 'baseline', '1', '2')
#count_group_cont(tbl_thresh_3_dic, t1_thresh_3_dic, 'baseline', '1', '3')
count_group_cont(tbl_thresh_4_dic, t1_thresh_4_dic, 'baseline', '1', '4')
# %%
#test group continuity between follow-up 1 and follow-up 2
#count_group_cont(t1_thresh_2_dic, t2_thresh_2_dic, '1', '2', '2')
#count_group_cont(t1_thresh_3_dic, t2_thresh_3_dic, '1', '2', '3')
count_group_cont(t1_thresh_4_dic, t2_thresh_4_dic, '1', '2', '4')

# %%
#test group continuity between follow-up 2 and follow-up 3
#count_group_cont(t2_thresh_2_dic, t3_thresh_2_dic, '2', '3', '2')
#count_group_cont(t2_thresh_3_dic, t3_thresh_3_dic, '2', '3', '3')
count_group_cont(t2_thresh_4_dic, t3_thresh_4_dic, '2', '3', '4')
# %%
#create lists of ABCD IDs that fit into HR groups for endorsed bother thresholds 2, 3, and 4
pps_hr_2 = pqbc_totals
pps_hr_3 = pqbc_totals
pps_hr_4 = pqbc_totals
thresh_2_list = []
thresh_3_list = []
thresh_4_list = []
for row in pqbc_2.index:
    if pqbc_2['pqbc_bother_totals'][row] >= 2:
        thresh_2_list.append(pqbc_2['subjectkey'][row])
    if pqbc_2['pqbc_bother_totals'][row] >= 3:
        thresh_3_list.append(pqbc_2['subjectkey'][row])
    if pqbc_2['pqbc_bother_totals'][row] >= 4:
        thresh_4_list.append(pqbc_2['subjectkey'][row])

# %%
#create HR group column for threshold 2
pps_hr_2['group'] = 0
for row in pps_hr_2.index:
    if pps_hr_2['subjectkey'][row] in thresh_2_list:
        pps_hr_2['group'][row] = 1
pps_hr_2.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PPS_HR_T2_thresh_2.csv')

# %%
#create HR group column for threshold 2
pps_hr_3['group'] = 0
for row in pps_hr_3.index:
    if pps_hr_3['subjectkey'][row] in thresh_3_list:
        pps_hr_3['group'][row] = 1
pps_hr_3.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PPS_HR_T2_thresh_3.csv')
        
# %%
#create HR group column for threshold 2
pps_hr_4['group'] = 0
for row in pps_hr_4.index:
    if pps_hr_4['subjectkey'][row] in thresh_4_list:
        pps_hr_4['group'][row] = 1
pps_hr_4.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PPS_HR_T2_thresh_4.csv')

# %%
