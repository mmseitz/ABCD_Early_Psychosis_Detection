
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

#ABCD mental health summary scores (youth survey)
mhss = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_mhy02.csv')

#PQ-BC HR/LR groups (High risk group defined as endorsing being bothered by 3+ PQ-BC questions)
pqbc_3 = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/PPS_HR_T2_thresh_3.csv')
pqbc_3 = pqbc_3.drop(columns= ['Unnamed: 0'])

#ABCD Child Behavioral Checklist (CBCL)
cbcl = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/ABCD_CBCL_followup2.csv')

#ABCD Kiddie Schedule for Affective Disorders and Schizophrenia 5 (Youth Questionnaire)
ksads_y = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/abcd_ksad501.csv')

#ABCD ABCD Kiddie Schedule for Affective Disorders and Schizophrenia 5 (Parent Questionnaire)
ksads_p = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/abcd_ksad01.csv')

#ABCD Summary Scores Medical History
med_ss = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/abcd_medhxss01.csv')

#ABCD Parent Family History Summary Scores (does not contain family historyof psychosis information)
fam_ss = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/abcd_fhxssp01.csv')

#ABCD Summary Scores Developmental History
dev_ss = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/abcd_devhxss01.csv')

#ABCD Developmental History Questionniare
devhx = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/dhx01.csv')

#ABCD Family History Questionnaire (contains information on family history of psychosis)
famhx = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/fhxp201.csv')
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
#Preprocess mental health summary scores df
#select salient columns (mean etc. columns omitted due to high NaN frequency)
#Columns pertaining to ACEs, UPPS-P, BIS/BAS, delinquency, 7UP, peer experiences (nts--read publications cited in NIH query)
mhss = mhss[['subjectkey', 'eventname', 'interview_age', 'ple_y_ss_total_number', 'ple_y_ss_total_good', 'ple_y_ss_total_bad', 'ple_y_ss_affect_sum', 
             'ple_y_ss_affected_good_sum', 'ple_y_ss_affected_bad_sum', 'upps_y_ss_negative_urgency', 'upps_y_ss_lack_of_planning', 
             'upps_y_ss_sensation_seeking', 'upps_y_ss_positive_urgency', 'upps_y_ss_lack_of_perseverance', 'bis_y_ss_bis_sum', 'bis_y_ss_bas_rr', 
             'bis_y_ss_bas_drive', 'bis_y_ss_bas_fs', 'delq_y_ss_sum', 'sup_y_ss_sum', 'gish_y_ss_m_sum', 'gish_y_ss_f_sum', 
              'peq_ss_relational_victim', 'peq_ss_reputation_aggs', 'peq_ss_reputation_victim', 'peq_ss_overt_aggression', 
              'peq_ss_overt_victim', 'peq_ss_relational_aggs']]
#['delq_y_ss_sum', 'sup_y_ss_sum'] dropped due to high NaN frequency
mhss = mhss.drop(columns= ['delq_y_ss_sum', 'sup_y_ss_sum'])
#Isolate follow-up 2 data
mhss = mhss.loc[mhss['eventname'] == '2_year_follow_up_y_arm_1']
#Count NaNs
count_nans(mhss) #max Nan 268 (not including sex-divided Gish q's ['gish_y_ss_m_sum', 'gish_y_ss_f_sum'])
#save CSV
#mhss.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/mhss_reduced.csv')
# %%
####
#Preprocess CBCL df
#rename subject column
cbcl = cbcl.rename(columns = {'EID': 'subjectkey'})
#select salient columns (thought score and total score)--analyze t-score
cbcl = cbcl[['subjectkey', 'Interview', 'cbcl_scr_syn_thought_r', 'cbcl_scr_syn_thought_t', 'cbcl_scr_syn_external_r', 'cbcl_scr_syn_external_t', 'cbcl_scr_syn_internal_r', 
             'cbcl_scr_syn_internal_t', 'cbcl_scr_syn_totprob_r', 'cbcl_scr_syn_totprob_t']]
#count NaNs
count_nans(cbcl) #df shape (6552, 10) 
#save csv
#cbcl.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/cbcl_reduced.csv')
# %%
####
#Preprocess KSADS-5 (Youth Questionnaire) csv
#all psychosis-related questions from all timepoints contain value '555' indicating that these questions were not administered at any ABCD wave
# %%
####
#Preprocess KSADS-5 (Parent Questionnaire) csv
#select data from follow up 2 only
k_p = ksads_p
k_p = k_p.loc[k_p['eventname'] == '2_year_follow_up_y_arm_1']
#count NaNs in each column
count_nans(k_p) #<50 NaNs per column; df shape (6571, 964)
#select relevant columns--pertaining to psycghosis symptoms/diagnoses (non-drug use related): (hallucinations [past/present], 
#delusions [past/present], associated psychotic symptoms [past/present], diagnoses hallucinations [past/present], diagnoses delusions [past/present],
# diagnosis associated psychotic sympotms [past/present], diagnosis unspecified psychotic disorder [past/present])
k_sx = ['ksads_4_16_p', 'ksads_4_17_p', 'ksads_4_236_p', 'ksads_4_235_p', 'ksads_4_232_p', 'ksads_4_233_p', 'ksads_4_231_p', 'ksads_4_249_p', 
        'ksads_4_19_p', 'ksads_4_248_p', 'ksads_4_18_p', 'ksads_4_258_p', 'ksads_4_259_p']
k_dx = ['ksads_4_826_p', 'ksads_4_827_p', 'ksads_4_829_p', 'ksads_4_828_p', 'ksads_4_851_p', 'ksads_4_852_p', 'ksads_4_850_p', 'ksads_4_849_p']
#create list of all relevant columns
kp_cols = ['subjectkey', 'eventname', 'sex', 'ksads_4_16_p', 'ksads_4_17_p', 'ksads_4_236_p', 'ksads_4_235_p', 'ksads_4_232_p', 'ksads_4_233_p', 'ksads_4_231_p', 'ksads_4_249_p', 
        'ksads_4_19_p', 'ksads_4_248_p', 'ksads_4_18_p', 'ksads_4_258_p', 'ksads_4_259_p', 'ksads_4_826_p', 'ksads_4_827_p', 'ksads_4_829_p', 'ksads_4_828_p', 'ksads_4_851_p', 
        'ksads_4_852_p', 'ksads_4_850_p', 'ksads_4_849_p']
#create df that contains only psychosis-related columns
k_psych = k_p[kp_cols]
#change object to float in sx and dx columns
to_num = ['ksads_4_16_p', 'ksads_4_17_p', 'ksads_4_236_p', 'ksads_4_235_p', 'ksads_4_232_p', 'ksads_4_233_p', 'ksads_4_231_p', 'ksads_4_249_p', 'ksads_4_19_p',
        'ksads_4_248_p', 'ksads_4_18_p', 'ksads_4_258_p', 'ksads_4_259_p', 'ksads_4_826_p', 'ksads_4_827_p', 'ksads_4_829_p', 'ksads_4_828_p', 'ksads_4_851_p', 
        'ksads_4_852_p', 'ksads_4_850_p', 'ksads_4_849_p']
k_psych_num = k_psych[to_num].apply(pd.to_numeric)
#change 888 (questions intentionally omitted due to modular nature of KSADS interview) into 0
for col in to_num:
    k_psych_num.loc[k_psych_num[col] == 888, [col]] = 0
#Tally total sx and dx in new columns
k_psych_num['k_p_total_psychosis_sx'] = k_psych_num[k_sx].sum(axis=1) 
k_psych_num['k_p_total_psychosis_dx'] = k_psych_num[k_dx].sum(axis=1)
#add tallied columns to main df
k_psych['k_p_total_psychosis_sx'] = k_psych_num['k_p_total_psychosis_sx']
k_psych['k_p_total_psychosis_dx'] = k_psych_num['k_p_total_psychosis_dx']
#drop 555 rows (participants for which questionnaire was not administered at this wave)
for row in k_psych.index:
    if k_psych['k_p_total_psychosis_sx'][row] > 100:
        k_psych = k_psych.drop([row])
#Create df that contains only identification info and tallied totals
k_psych_small = k_psych[['subjectkey', 'eventname', 'sex', 'k_p_total_psychosis_sx', 'k_p_total_psychosis_dx']]
#save small csv to folder
k_psych_small.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/ksads_parent_short.csv')
# %%
####
#Preprocess medical history csv
#create new column to tally incidents
med = med_ss
med = med.drop([0])
mhx_cols = []
all_cols = list(med)
for i in all_cols: 
    if 'medhx_' in i:
        mhx_cols.append(i)
med.dtypes
med_num = med[mhx_cols].apply(pd.to_numeric)
med_num['injury_totals'] = med_num[mhx_cols].sum(axis=1)
#add tallied column to main datasheet
med['injury_totals'] = med_num['injury_totals']
#select salient columns for reduced csv
med_small = med[['subjectkey', 'eventname', 'sex', 'injury_totals']]
#save reduced csv to folder
med_small.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/med_hx_reduced.csv')
# %%
####
#Preprocess Developmental History Summary Scores csv
#Drop first row (contains text)
dev = dev_ss.drop([0])
#identify columns that indicate presence or absence of given substance use
drug_cols = ['devhx_ss_8_cigs_per_day_p', 'devhx_ss_8_marijuana_amt_p', 'devhx_ss_8_coc_crack_amt_p', 'devhx_ss_8_her_morph_amt_p', 
             'devhx_ss_8_oxycont_amt_p']
drugs = dev[drug_cols]
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
dev['total_substances_used'] = drugs['total_substances_used']
#Select salient columns for reduced csv (total substances ued per pregnancy, how many weeks premature birth)
dev_small = dev[['subjectkey', 'eventname', 'sex', 'total_substances_used', 'devhx_ss_12_p']]
#identify whether participants were born preterm (defined as born 3+ weeks prematurely)
dev_small['preterm'] = 0 
dev_small['devhx_ss_12_p'] = pd.to_numeric(dev_small['devhx_ss_12_p'])
dev_small = dev_small.rename(columns={'devhx_ss_12_p':'weeks_preterm'})
for row in dev_small.index: 
    if dev_small['weeks_preterm'][row] >= 3: 
        dev_small['preterm'][row] = 1
# %%
####
#Preprocess developmental history questionnaire
#drop first row (containing text)
d_hx = devhx.drop([0])
#identify columns containing information regarding developmental milestones
milestones = ['subjectkey', 'visit', 'sex', 'devhx_19a_p', 'devhx_19b_p', 'devhx_19c_p', 'devhx_19d_p', 'devhx_20_p', 'devhx_21_p', 'devhx_22_3_p', 'devhx_23b_p']
d_hx = d_hx[milestones]
#convert '999' to NaN 
d_hx['devhx_19a_p'] = pd.to_numeric(d_hx['devhx_19a_p'])
d_hx.loc[d_hx['devhx_19a_p'] == 999, ['devhx_19a_p']] = 'NaN'
d_hx['devhx_19b_p'] = pd.to_numeric(d_hx['devhx_19b_p'])
d_hx.loc[d_hx['devhx_19b_p'] == 999, ['devhx_19b_p']] = 'NaN'
d_hx['devhx_19c_p'] = pd.to_numeric(d_hx['devhx_19c_p'])
d_hx.loc[d_hx['devhx_19c_p'] == 999, ['devhx_19c_p']] = 'NaN'
d_hx['devhx_19d_p'] = pd.to_numeric(d_hx['devhx_19d_p'])
d_hx.loc[d_hx['devhx_19d_p'] == 999, ['devhx_19d_p']] = 'NaN'
d_hx['devhx_20_p'] = pd.to_numeric(d_hx['devhx_20_p'])
d_hx.loc[d_hx['devhx_20_p'] == 999, ['devhx_20_p']] = 'NaN'
d_hx['devhx_21_p'] = pd.to_numeric(d_hx['devhx_21_p'])
d_hx.loc[d_hx['devhx_21_p'] == 999, ['devhx_21_p']] = 'NaN'
d_hx['devhx_22_3_p'] = pd.to_numeric(d_hx['devhx_22_3_p'])
d_hx.loc[d_hx['devhx_22_3_p'] == 999, ['devhx_22_3_p']] = 'NaN'
d_hx['devhx_23b_p'] = pd.to_numeric(d_hx['devhx_23b_p'])
d_hx.loc[d_hx['devhx_23b_p'] == 999, ['devhx_23b_p']] = 'NaN'
#rename columns
d_hx = d_hx.rename(columns= {'devhx_19a_p': 'age_roll_over', 'devhx_19b_p': 'age_sit', 'devhx_19c_p': 'age_walk', 
                             'devhx_19d_p': 'age_first_word', 'devhx_20_p': 'motor_dev_comparison', 'devhx_21_p': 
                             'speech_dev_comparison', 'devhx_22_3_p': 'wet_bed', 'devhx_23b_p': 'age_stop_bedwetting'})
d_hx = d_hx.drop(columns= ['visit'])
#merge developmental summary scores and developmental questionnaire dfs
dev_data = pd.merge(dev_small, d_hx, on = ['subjectkey', 'sex'])
#save csv
dev_data.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/devhx_reduced.csv')
# %%
####
#Preprocess family history questionnaire (part 2; part 1 contains no psychosis-related questions)
#drop first row (contains text)
f_hx = famhx.drop([0])
#identify questions regarding family history of psychosis/psychosis symptoms (first-degree relatives and biological grandparents)
fam_qs = ['subjectkey', 'visit', 'sex', 'fam_history_8_yes_no', 'fam_history_q8a_visions', 'fam_history_q8d_visions', 'q8k_full_sib_young1_visions', 'q8k_full_sib_young2_visions', 
          'q8k_full_sib_young3_visions', 'q8k_full_sib_young4_visions', 'q8k_full_sib_young5_visions', 'q8l_full_sib_old1_visions', 
          'q8l_full_sib_old2_visions', 'q8l_full_sib_old3_visions', 'q8l_full_sib_old4_visions', 'q8l_full_sib_old5_visions', 'fam_history_q8b_visions', 
          'fam_history_q8c_visions', 'fam_history_q8e_visions', 'fam_history_q8f_visions']
f_hx = f_hx[fam_qs]
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
famhx_short = f_hx[['subjectkey', 'visit', 'sex', 'first_deg_psychosis_hx', 'grandparent_psychosis_hx']]
#save shortened family history csv
famhx_short.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/famhx_short.csv')
# %%
