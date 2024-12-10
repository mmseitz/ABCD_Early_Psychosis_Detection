
#This script preprocesses CSVs pertaining to participants' environment, mental health, and demographics
#CSVs: pdem02, mhy02, abcd_cbcls01, ksads2daic_use_only01, ksads2daic_use_only_p01, abcd_medhxss01, abcd_fhxssp01, abcd_devhxss01, dhx01, fhxp201

# %%
####
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np

# %%
####
#import CSVs

#demographics (preprocessed CSV with select variables--income/race/education/sex)
dems = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/reduced_dem.csv')

#ABCD Child Behavioral Checklist (CBCL)
cbcl = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_p_cbcl.csv')

#ABCD Kiddie Schedule for Affective Disorders and Schizophrenia 5 (Youth Questionnaire)
ksads_y = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_y_ksads_ss.csv')

#ABCD ABCD Kiddie Schedule for Affective Disorders and Schizophrenia 5 (Parent Questionnaire)
ksads_p = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_p_ksads_ss.csv')

#ABCD Medical History
medhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/ph_p_mhx.csv')

#ABCD Developmental History Questionniare
devhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/ph_p_dhx.csv')

#ABCD Family History Questionnaire (contains information on family history of psychosis)
famhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_p_fhx.csv')

#Adverse Life Expereinces Questionnaire (Parent)
ale_p = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_p_le.csv')

#Adverse Life Experiences Questionnaire (Youth)
ale_y = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_y_le.csv')

#Peer Experiences Questionnaire (Youth)
peq = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_y_peq.csv')

#Neighborhood Safety Questionnaire (Youth)
nsc = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/ce_y_nsc.csv')

#Discrimination Questionnaire (Youth)
disc_y = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/ce_y_dm.csv')

#Neglectful Behavior Scale (Youth)
mnbs = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/ce_y_mnbs.csv')

#Family Environment Scale
fes = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/ce_y_fes.csv') 

#Community Cohesion Scale
ccs = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/ce_p_comc.csv')
# %%
####
#function to check how many NaNs exist in each selected df column
def count_nans(df):
    l = list(df)
    for i in l:
        if df[i].isna().sum() >= 50:
            print(i, df[i].isna().sum())
# %%
####
#Preprocess demographics CVS
#convert variables to numbers for easier analysis
dems = dems.drop(columns = ['Unnamed: 0'])
dems = dems.rename(columns= {'demo_sex_v2': 'sex'})
dems.loc[dems['p_edu'] == 'A. high school', 'p_edu'] = 1
dems.loc[dems['p_edu'] == 'B. college', 'p_edu'] = 2
dems.loc[dems['p_edu'] == 'C. post grad', 'p_edu'] = 3
dems.loc[dems['income'] == 'A.lower', 'income'] = 1
dems.loc[dems['income'] == 'B.middle', 'income'] = 2
dems.loc[dems['income'] == 'C.higher', 'income'] = 3
dems.loc[dems['race'] == 'A.white', 'race'] = 1
dems.loc[dems['race'] == 'B.black', 'race'] = 2
dems.loc[dems['race'] == 'C.hispanic', 'race'] = 3
dems.loc[dems['race'] == 'D.Other', 'race'] = 4
#dems.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/num_demos.csv')
# %%
####
#Preprocess CBCL df
#select salient columns (thought score and total score)--analyze t-score
cbcl = cbcl[['src_subject_id', 'eventname', 'cbcl_scr_syn_thought_r', 'cbcl_scr_syn_thought_t', 'cbcl_scr_syn_external_r', 'cbcl_scr_syn_external_t', 'cbcl_scr_syn_internal_r', 
             'cbcl_scr_syn_internal_t', 'cbcl_scr_syn_totprob_r', 'cbcl_scr_syn_totprob_t']]
#count NaNs
count_nans(cbcl) 

#make df containing data only from T2
cbcl_2 = cbcl.loc[cbcl['eventname'] == '2_year_follow_up_y_arm_1']

#make df containing data only from T4
cbcl_4 = cbcl.loc[cbcl['eventname'] == '4_year_follow_up_y_arm_1']
#save csvs
cbcl_2.to_csv('/Users/madeleineseitz/Desktop/abcd5_cbcl_T2_reduced.csv')
cbcl_4.to_csv('/Users/madeleineseitz/Desktop/abcd5_cbcl_T4_reduced.csv')
# %%
####
#Preprocess KSADS-5 (Youth Questionnaire) csv
#all psychosis-related questions from all timepoints contain value '555' indicating that these questions were not administered at any ABCD wave
#ksads_y_4 = ksads_y.loc[ksads_y['eventname'] == '4_year_follow_up_y_arm_1']
# %%
####
#Preprocess KSADS-5 (Parent Questionnaire) csv
#select data from follow up 2 only
k_p = ksads_p
k_p_2 = k_p.loc[k_p['eventname'] == '2_year_follow_up_y_arm_1']
k_p_4 = k_p.loc[k_p['eventname'] == '4_year_follow_up_y_arm_1']
#count NaNs in each column
count_nans(k_p) #<50 NaNs per column; df shape (6571, 964)
#select relevant columns--pertaining to psycghosis symptoms/diagnoses (non-drug use related): (hallucinations [past/present], 
#delusions [past/present], associated psychotic symptoms [past/present], diagnoses hallucinations [past/present], diagnoses delusions [past/present],
# diagnosis associated psychotic sympotms [past/present], diagnosis unspecified psychotic disorder [past/present])
k_sx = ['ksads_4_16_p', 'ksads_4_17_p', 'ksads_4_236_p', 'ksads_4_235_p', 'ksads_4_232_p', 'ksads_4_233_p', 'ksads_4_231_p', 'ksads_4_249_p', 
        'ksads_4_19_p', 'ksads_4_248_p', 'ksads_4_18_p', 'ksads_4_258_p', 'ksads_4_259_p']
k_dx = ['ksads_4_826_p', 'ksads_4_827_p', 'ksads_4_829_p', 'ksads_4_828_p', 'ksads_4_851_p', 'ksads_4_852_p', 'ksads_4_850_p', 'ksads_4_849_p']
#create list of all relevant columns
kp_cols = ['src_subject_id', 'eventname', 'ksads_4_16_p', 'ksads_4_17_p', 'ksads_4_236_p', 'ksads_4_235_p', 'ksads_4_232_p', 'ksads_4_233_p', 'ksads_4_231_p', 'ksads_4_249_p', 
        'ksads_4_19_p', 'ksads_4_248_p', 'ksads_4_18_p', 'ksads_4_258_p', 'ksads_4_259_p', 'ksads_4_826_p', 'ksads_4_827_p', 'ksads_4_829_p', 'ksads_4_828_p', 'ksads_4_851_p', 
        'ksads_4_852_p', 'ksads_4_850_p', 'ksads_4_849_p']
#create df that contains only psychosis-related columns
k_psych_2 = k_p_2[kp_cols]
k_psych_4 = k_p_4[kp_cols]
#change object to float in sx and dx columns
to_num = ['ksads_4_16_p', 'ksads_4_17_p', 'ksads_4_236_p', 'ksads_4_235_p', 'ksads_4_232_p', 'ksads_4_233_p', 'ksads_4_231_p', 'ksads_4_249_p', 'ksads_4_19_p',
        'ksads_4_248_p', 'ksads_4_18_p', 'ksads_4_258_p', 'ksads_4_259_p', 'ksads_4_826_p', 'ksads_4_827_p', 'ksads_4_829_p', 'ksads_4_828_p', 'ksads_4_851_p', 
        'ksads_4_852_p', 'ksads_4_850_p', 'ksads_4_849_p']
k_psych_2_num = k_psych_2[to_num].apply(pd.to_numeric)
k_psych_4_num = k_psych_4[to_num].apply(pd.to_numeric)
#change 888 (questions intentionally omitted due to modular nature of KSADS interview) into 0
for col in to_num:
    k_psych_2_num.loc[k_psych_2_num[col] == 888, [col]] = 0
    k_psych_4_num.loc[k_psych_4_num[col] == 888, [col]] = 0
#Tally total sx and dx in new columns
k_psych_2_num['k_p_total_psychosis_sx'] = k_psych_2_num[k_sx].sum(axis=1) 
k_psych_2_num['k_p_total_psychosis_dx'] = k_psych_2_num[k_dx].sum(axis=1)

k_psych_4_num['k_p_total_psychosis_sx'] = k_psych_4_num[k_sx].sum(axis=1) 
k_psych_4_num['k_p_total_psychosis_dx'] = k_psych_4_num[k_dx].sum(axis=1)
#add tallied columns to main df
k_psych_2['k_p_total_psychosis_sx'] = k_psych_2_num['k_p_total_psychosis_sx']
k_psych_2['k_p_total_psychosis_dx'] = k_psych_2_num['k_p_total_psychosis_dx']

k_psych_4['k_p_total_psychosis_sx'] = k_psych_4_num['k_p_total_psychosis_sx']
k_psych_4['k_p_total_psychosis_dx'] = k_psych_4_num['k_p_total_psychosis_dx']
#drop 555 rows (participants for which questionnaire was not administered at this wave)
for row in k_psych_2.index:
    if k_psych_2['k_p_total_psychosis_sx'][row] > 100:
        k_psych_2 = k_psych_2.drop([row])

for row in k_psych_4.index:
    if k_psych_4['k_p_total_psychosis_sx'][row] > 100:
        k_psych_4 = k_psych_4.drop([row])
#Create df that contains only identification info and tallied totals
k_psych_2_small = k_psych_2[['src_subject_id', 'eventname', 'k_p_total_psychosis_sx', 'k_p_total_psychosis_dx']]

k_psych_4_small = k_psych_4[['src_subject_id', 'eventname', 'k_p_total_psychosis_sx', 'k_p_total_psychosis_dx']]
#save small csv to folder
k_psych_2_small.to_csv('/Users/madeleineseitz/Desktop/abcd_5_ksads_parent_T2_short.csv')

#k_psych_4_small.to_csv('/Users/madeleineseitz/Desktop/abcd_5_ksads_parent_T4_short.csv')
#no responses for psychosis questions in T4
# %%
####
#Preprocess medical history csv
#create new column to tally incidents
#data only collected at baseline
med = medhx
med_ss_cols = ['src_subject_id', 'eventname', 'medhx_ss_4b_p', 'medhx_ss_5b_p', 'medhx_ss_6a_times_p', 'medhx_ss_6b_times_p', 'medhx_ss_6c_times_p', 'medhx_ss_6d_times_p', 
               'medhx_ss_6e_times_p', 'medhx_ss_6f_times_p', 'medhx_ss_6g_times_p', 'medhx_ss_6h_times_p', 'medhx_ss_6i_times_p', 
               'medhx_ss_6j_times_p', 'medhx_ss_6k_times_p', 'medhx_ss_6l_times_p', 'medhx_ss_6m_times_p', 'medhx_ss_6n_times_p', 'medhx_ss_6o_times_p', 
               'medhx_ss_6p_times_p', 'medhx_ss_6q_times_p', 'medhx_ss_6r_times_p', 'medhx_ss_6s_times_p']
to_num = ['total_times_to_ER', 'total_times_to_ER', 'medhx_ss_6a_times_p', 'medhx_ss_6b_times_p', 'medhx_ss_6c_times_p', 'medhx_ss_6d_times_p', 
               'medhx_ss_6e_times_p', 'medhx_ss_6f_times_p', 'medhx_ss_6g_times_p', 'medhx_ss_6h_times_p', 'medhx_ss_6i_times_p', 
               'medhx_ss_6j_times_p', 'medhx_ss_6k_times_p', 'medhx_ss_6l_times_p', 'medhx_ss_6m_times_p', 'medhx_ss_6n_times_p', 'medhx_ss_6o_times_p', 
               'medhx_ss_6p_times_p', 'medhx_ss_6q_times_p', 'medhx_ss_6r_times_p', 'medhx_ss_6s_times_p']
injuries = ['medhx_ss_6a_times_p', 'medhx_ss_6b_times_p', 'medhx_ss_6c_times_p', 'medhx_ss_6d_times_p', 
               'medhx_ss_6e_times_p', 'medhx_ss_6f_times_p', 'medhx_ss_6g_times_p', 'medhx_ss_6h_times_p', 'medhx_ss_6i_times_p', 
               'medhx_ss_6j_times_p', 'medhx_ss_6k_times_p', 'medhx_ss_6l_times_p', 'medhx_ss_6m_times_p', 'medhx_ss_6n_times_p', 'medhx_ss_6o_times_p', 
               'medhx_ss_6p_times_p', 'medhx_ss_6q_times_p', 'medhx_ss_6r_times_p', 'medhx_ss_6s_times_p']
#select sum score columns from df
med_ss = med[med_ss_cols]
#rename ER cols
med_ss = med_ss.rename(columns= {'medhx_ss_4b_p': 'total_times_to_ER', 'medhx_ss_5b_p': 'times_to_ER_past_year'})
#separate into T2 and T4 timepoints
med_ss = med_ss.loc[med_ss['eventname'] == 'baseline_year_1_arm_1']
med_ss.dtypes
#convert relevant columns to numeric
med_num = med_ss[to_num].apply(pd.to_numeric)
#sum across injurie ss columns
med_num['injury_totals'] = med_num[injuries].sum(axis=1)
#add sum col to med_ss
med_ss['injury_totals'] = med_num['injury_totals']

#select salient columns for reduced csv
med_small = med_ss[['src_subject_id', 'eventname', 'total_times_to_ER', 'times_to_ER_past_year', 'injury_totals']]

#save reduced csv to folder
med_small.to_csv('/Users/madeleineseitz/Desktop/abcd5_med_hx_reduced.csv')
# %%
####
#Preprocess Developmental History Summary Scores csv
#isolate baseline data
devhx = devhx.loc[devhx['eventname'] == 'baseline_year_1_arm_1']
#identify columns that indicate presence or absence of given substance use
drug_cols = ['devhx_ss_8_cigs_per_day_p', 'devhx_ss_8_marijuana_amt_p', 'devhx_ss_8_coc_crack_amt_p', 'devhx_ss_8_her_morph_amt_p', 
             'devhx_ss_8_oxycont_amt_p']
drugs = devhx[drug_cols]
#convert object to float
drugs = drugs.apply(pd.to_numeric)
#fill NaN values with 0
drugs = drugs.fillna(0)
#replace frequency of use with '1' to indicate use (all substances weighted equally; cursory analysis concerned with only presence or absence)
drugs.loc[drugs['devhx_ss_8_cigs_per_day_p'] != 0, ['devhx_ss_8_cigs_per_day_p']] = 1
drugs.loc[drugs['devhx_ss_8_marijuana_amt_p'] != 0, ['devhx_ss_8_marijuana_amt_p']] = 1
drugs.loc[drugs['devhx_ss_8_coc_crack_amt_p'] != 0, ['devhx_ss_8_coc_crack_amt_p']] = 1
drugs.loc[drugs['devhx_ss_8_her_morph_amt_p'] != 0, ['devhx_ss_8_her_morph_amt_p']] = 1
drugs.loc[drugs['devhx_ss_8_oxycont_amt_p'] != 0, ['devhx_ss_8_oxycont_amt_p']] = 1
#tally total number of substances used per pregnancy
drugs['total_substances_used'] = drugs.sum(axis=1)
#map new column to main df
devhx['total_substances_used'] = drugs['total_substances_used']
#Select salient columns for reduced csv (total substances ued per pregnancy, how many weeks premature birth)
dev_small = devhx[['src_subject_id', 'eventname', 'total_substances_used', 'devhx_ss_12_p']]
#identify whether participants were born preterm (defined as born 3+ weeks prematurely)
dev_small['preterm'] = 0 
dev_small['devhx_ss_12_p'] = pd.to_numeric(dev_small['devhx_ss_12_p'])
dev_small.loc[dev_small['devhx_ss_12_p'] == 999, ['devhx_ss_12_p']] = 'NaN'
dev_small = dev_small.rename(columns={'devhx_ss_12_p':'weeks_preterm'})
for row in dev_small.index: 
    if dev_small['weeks_preterm'][row] != 'NaN':
        if dev_small['weeks_preterm'][row] >= 3: 
            dev_small['preterm'][row] = 1
#identify columns containing information regarding developmental milestones
milestones = ['src_subject_id', 'eventname', 'age_roll_over', 'age_sit', 'age_walk', 'age_first_word', 'motor_dev_comparison', 'speech_dev_comparison', 'wet_bed', 'age_stop_bedwetting']
#convert '999' to NaN 
devhx['devhx_19a_p'] = pd.to_numeric(devhx['devhx_19a_p'])
devhx.loc[devhx['devhx_19a_p'] == 999, ['devhx_19a_p']] = 'NaN'
devhx['devhx_19b_p'] = pd.to_numeric(devhx['devhx_19b_p'])
devhx.loc[devhx['devhx_19b_p'] == 999, ['devhx_19b_p']] = 'NaN'
devhx['devhx_19c_p'] = pd.to_numeric(devhx['devhx_19c_p'])
devhx.loc[devhx['devhx_19c_p'] == 999, ['devhx_19c_p']] = 'NaN'
devhx['devhx_19d_p'] = pd.to_numeric(devhx['devhx_19d_p'])
devhx.loc[devhx['devhx_19d_p'] == 999, ['devhx_19d_p']] = 'NaN'
devhx['devhx_20_p'] = pd.to_numeric(devhx['devhx_20_p'])
devhx.loc[devhx['devhx_20_p'] == 999, ['devhx_20_p']] = 'NaN'
devhx['devhx_21_p'] = pd.to_numeric(devhx['devhx_21_p'])
devhx.loc[devhx['devhx_21_p'] == 999, ['devhx_21_p']] = 'NaN'
devhx['devhx_22_3_p'] = pd.to_numeric(devhx['devhx_22_3_p'])
devhx.loc[devhx['devhx_22_3_p'] == 999, ['devhx_22_3_p']] = 'NaN'
devhx['devhx_23b_p'] = pd.to_numeric(devhx['devhx_23b_p'])
devhx.loc[devhx['devhx_23b_p'] == 999, ['devhx_23b_p']] = 'NaN'
#rename columns
devhx = devhx.rename(columns= {'devhx_19a_p': 'age_roll_over', 'devhx_19b_p': 'age_sit', 'devhx_19c_p': 'age_walk', 
                             'devhx_19d_p': 'age_first_word', 'devhx_20_p': 'motor_dev_comparison', 'devhx_21_p': 
                             'speech_dev_comparison', 'devhx_22_3_p': 'wet_bed', 'devhx_23b_p': 'age_stop_bedwetting'})
dev_m = devhx[milestones]
#merge developmental summary scores and developmental questionnaire dfs
dev_data = pd.merge(dev_small, dev_m, on = ['src_subject_id', 'eventname'])
#save csv
dev_data.to_csv('/Users/madeleineseitz/Desktop/abcd5_baseline_devhx_reduced.csv')
# %%
####
#Preprocess family history questionnaire (all data collected at baseline)
#identify questions regarding family history of psychosis/psychosis symptoms (first-degree relatives and biological grandparents)
fam_qs = ['src_subject_id', 'eventname', 'fam_history_8_yes_no', 'fam_history_q8a_visions', 'fam_history_q8d_visions', 'q8k_full_sib_young1_visions', 'q8k_full_sib_young2_visions', 
          'q8k_full_sib_young3_visions', 'q8k_full_sib_young4_visions', 'q8k_full_sib_young5_visions', 'q8l_full_sib_old1_visions', 
          'q8l_full_sib_old2_visions', 'q8l_full_sib_old3_visions', 'q8l_full_sib_old4_visions', 'q8l_full_sib_old5_visions', 'fam_history_q8b_visions', 
          'fam_history_q8c_visions', 'fam_history_q8e_visions', 'fam_history_q8f_visions']
f_hx = famhx[fam_qs]
#identify columns containing psychosis history of first-degree relatives
r_first = ['fam_history_q8a_visions', 'fam_history_q8d_visions', 'q8k_full_sib_young1_visions', 'q8k_full_sib_young2_visions', 
          'q8k_full_sib_young3_visions', 'q8k_full_sib_young4_visions', 'q8k_full_sib_young5_visions', 'q8l_full_sib_old1_visions', 
          'q8l_full_sib_old2_visions', 'q8l_full_sib_old3_visions', 'q8l_full_sib_old4_visions', 'q8l_full_sib_old5_visions']
#identify columns containing psychosis history of biological grandparents
r_other = ['fam_history_q8b_visions', 'fam_history_q8c_visions', 'fam_history_q8e_visions', 'fam_history_q8f_visions']
#make column values numeric
fam = f_hx[r_first]
fam = fam.apply(pd.to_numeric)
for col in list(fam):
    fam.loc[fam[col] == 999, [col]] = 0
#calculate how many first degree relatives have a history of psychosis/psychosis sx
fam['first_deg_psychosis_hx'] = fam.sum(axis=1)
#append column to f_hx df
f_hx['first_deg_psychosis_hx'] = fam['first_deg_psychosis_hx']
#change columns to numeric values
grandp = f_hx[r_other]
grandp = grandp.apply(pd.to_numeric)
for col in list(grandp):
    grandp.loc[grandp[col] == 999, [col]] = 0
#calculate how many biological grandparents have a history of psychosis/psychosis sx
grandp['grandparent_psychosis_hx'] = grandp.sum(axis=1)
#append column to f_hx
f_hx['grandparent_psychosis_hx'] = grandp['grandparent_psychosis_hx']
#shorten df to summed columns
famhx_short = f_hx[['src_subject_id', 'eventname', 'first_deg_psychosis_hx', 'grandparent_psychosis_hx']]
#save shortened family history csv
famhx_short.to_csv('/Users/madeleineseitz/Desktop/abcd5_famhx_short.csv')
# %%
#Preprocess parent ALEs
#identify sum score columns
ale_p_ss = ale_p[['src_subject_id', 'eventname', 'ple_p_ss_total_number', 'ple_p_ss_total_good', 'ple_p_ss_total_bad', 'ple_p_ss_affected_good_sum', 'ple_p_ss_affected_bad_sum']]
#identify stats from follow-up-2
ale_p_T2 = ale_p_ss.loc[ale_p_ss['eventname'] == '2_year_follow_up_y_arm_1']
ale_p_T4 = ale_p_ss.loc[ale_p_ss['eventname'] == '4_year_follow_up_y_arm_1']
#save parent ALE T2 and T4 csvs
ale_p_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_ALE_p_T2_short.csv')
ale_p_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_ALE_p_T4_short.csv')
# %%
#Preprocess youth ALEs
#identify sum score columns
ale_y_ss = ale_y[['src_subject_id', 'eventname', 'ple_y_ss_total_number', 'ple_y_ss_total_good', 'ple_y_ss_total_bad', 'ple_y_ss_affected_good_sum', 'ple_y_ss_affected_bad_sum']]
#identify stats from follow-up-2
ale_y_T2 = ale_y_ss.loc[ale_y_ss['eventname'] == '2_year_follow_up_y_arm_1']
ale_y_T4 = ale_y_ss.loc[ale_y_ss['eventname'] == '4_year_follow_up_y_arm_1']
#save youth ALE T2 and T4 csvs
ale_y_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_ALE_y_T2_short.csv')
ale_y_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_ALE_y_T4_short.csv')
# %%
#preprocess peer experiences questionnarie
#isolate summary scores (relational aggression, relational victim [being left out; excluded], reputational aggression [rumors, gossipping],
# reputation victimization [being the victim of gossip, rumors], overt aggression [chasing, threatening, hitting], overt victim [being the victim of chasing, threatening, hitting], 
# relational aggresion [leaving people out, exclusion], )
peq_ss = ['src_subject_id', 'eventname', 'peq_ss_relational_victim', 'peq_ss_reputation_aggs', 'peq_ss_reputation_victim', 'peq_ss_overt_aggression', 'peq_ss_overt_victim', 
          'peq_ss_relational_aggs']
peq_ss = peq[peq_ss]
#isolate T2 and T4 scores
peq_ss_T2 = peq_ss.loc[peq['eventname'] == '2_year_follow_up_y_arm_1']
peq_ss_T4 = peq_ss.loc[peq['eventname'] == '4_year_follow_up_y_arm_1']
#save T2 and T4 csvs
peq_ss_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_PEQ_y_T2_ss.csv')
peq_ss_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_PEQ_y_T4_ss.csv')
# %%
#Preprocess neighborhood safety questioinnaire
#Question "my neighborhood is safe from crime" is only question selected--others omitted due to high NaN frequencies
nsc_qs = ['src_subject_id', 'eventname', 'neighborhood_crime_y']
nsc_short = nsc[nsc_qs]
#isolate T2 and T4 scores
nsc_T2 = nsc_short.loc[nsc_short['eventname'] == '2_year_follow_up_y_arm_1']
nsc_T4 = nsc_short.loc[nsc_short['eventname'] == '4_year_follow_up_y_arm_1']
#save T2 and T4 csvs
nsc_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_NSC_T2_short.csv')
nsc_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_NSC_T4_short.csv')
# %%
#Preprocess discrimination questionnaire
disc_cols = ['src_subject_id', 'eventname', 'dim_yesno_q1', 'dim_yesno_q2', 'dim_yesno_q3', 'dim_yesno_q4', 'dim_y_ss_mean']
d_qs = ['dim_yesno_q1', 'dim_yesno_q2', 'dim_yesno_q3', 'dim_yesno_q4', 'dim_y_ss_mean']
disc_short = disc_y[disc_cols]
#make sure columns are numeric; turn 777 (don't know) into 0 
for i in d_qs: 
    disc_short[i] = pd.to_numeric(disc_short[i])
    disc_short.loc[disc_short[i] == 999, [i]] = 'NaN'
    disc_short.loc[disc_short[i] == 777, [i]] = 0
#isolate T2 and T4 scores
disc_T2 = disc_short.loc[disc_short['eventname'] == '2_year_follow_up_y_arm_1']
disc_T4 = disc_short.loc[disc_short['eventname'] == '4_year_follow_up_y_arm_1']
#save T2 and T4 csvs
disc_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_disc_y_T2_short.csv')
disc_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_disc_y_T4_short.csv')
# %%
#Preprocessing multidimensional neglectful behavior scale
#all data collected at follow-up 3
#make sure columns are numeric; turn 777 (refuse to answer) into NaN
mnbs_small = mnbs.drop(columns= ['mnbs_admin'])
mnbs_cols = ['mnbs_bad', 'mnbs_doing', 'mnbs_friend', 'mnbs_homework', 'mnbs_play', 
             'mnbs_school', 'mnbs_trouble', 'mnbs_understand', 'mnbs_ss_monitor_supervision', 'mnbs_ss_mean_all', 'mnbs_ss_ed_support']
for i in mnbs_cols: 
    mnbs_small[i] = pd.to_numeric(mnbs_small[i])
    mnbs_small.loc[mnbs_small[i] == 777, [i]] = 'NaN'
#save neglect csv
mnbs_small.to_csv('/Users/madeleineseitz/Desktop/abcd5_mnbs_y_T3_short.csv')
# %%
#Preprocessing community cohesion scale
#pulling out sum scores
ccs_cols = ['src_subject_id', 'eventname', 'comc_ss_cohesion_p', 'comc_ss_control_p', 'comc_ss_collective_capacity_p', ]
ss_cols = ['comc_ss_cohesion_p', 'comc_ss_control_p', 'comc_ss_collective_capacity_p']
ccs_small = ccs[ccs_cols]
for i in ss_cols: 
    ccs_small[i] = pd.to_numeric(ccs_small[i])
ccs_T2 = ccs_small.loc[ccs_small['eventname'] == '2_year_follow_up_y_arm_1']
ccs_T4 = ccs_small.loc[ccs_small['eventname'] == '4_year_follow_up_y_arm_1']
#save to csv
ccs_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_ccs_p_short.csv')
ccs_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_ccs_p_short.csv')
# %%