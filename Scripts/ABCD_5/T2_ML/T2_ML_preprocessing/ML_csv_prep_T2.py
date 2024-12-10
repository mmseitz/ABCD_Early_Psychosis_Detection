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
#Import raw data
#Demographics
grouped_demos = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/matched_refined_groupings/T2_matched_demos.csv')
grouped_demos = pd.DataFrame(grouped_demos)
grouped_demos = grouped_demos.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
#Structural data
anat = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/sMRI/abcd5_T2_structural_data.csv')
anat = pd.DataFrame(anat)
anat = anat.drop(['Unnamed: 0'], axis=1)
#Cognitive Task Data
#Monetary Incentive Delay data
#Behavioral data
mid_behav = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_mid_behav_reduced.csv')
mid_behav = pd.DataFrame(mid_behav)
mid_behav = mid_behav.drop(['Unnamed: 0'], axis=1)
#Brain data
mid_brain = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_mid_mri_T2.csv')
mid_brain = pd.DataFrame(mid_brain)
mid_brain = mid_brain.drop(['Unnamed: 0'], axis=1)
#Stop Signal Task data
#Behavioral data
sst_behav = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_sst_beh_reduced.csv')
sst_behav = pd.DataFrame(sst_behav)
sst_behav = sst_behav.drop(['Unnamed: 0'], axis=1)
#Brain data
sst_brain = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_sst_brain_reduced.csv')
sst_brain = pd.DataFrame(sst_brain)
sst_brain = sst_brain.drop(['Unnamed: 0'], axis=1)
#Emotional N-Back data
#behavioral data
enb_behav = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_enb_behav_reduced.csv')
enb_behav = pd.DataFrame(enb_behav)
enb_behav = enb_behav.drop(['Unnamed: 0'], axis=1)
#brain data
enb_brain = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_enb_brain_reduced.csv')
enb_brain = pd.DataFrame(enb_brain)
enb_brain = enb_brain.drop(['Unnamed: 0'], axis=1)
#Game of Dice task
gdt = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_gdt_reduced.csv')
gdt = pd.DataFrame(gdt)
gdt = gdt.drop(['Unnamed: 0'], axis=1)
#Social Influence Task
sit = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_social_influence_reduced.csv')
sit = pd.DataFrame(sit)
sit = sit.drop(['Unnamed: 0'], axis=1)
#NIH Toolbox
nihtb = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_nihtb_reduced.csv')
nihtb = pd.DataFrame(nihtb)
nihtb = nihtb.drop(['Unnamed: 0'], axis=1)
#Little Man Task
lmt = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/tasks_preprocessed/T2/abcd5_T2_lmt_reduced.csv')
lmt = pd.DataFrame(lmt)
lmt = lmt.drop(['Unnamed: 0'], axis=1)
#Environment Data
#Developmental History
devhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_baseline_devhx_reduced.csv')
devhx = devhx.drop(['Unnamed: 0', 'eventname'], axis=1)
#Medical History
medhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_med_hx_reduced.csv')
medhx = medhx.drop(['Unnamed: 0', 'eventname'], axis=1)
#Adverse Life Experiences (Parent)
ale_p = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_ALE_p_T2_short.csv')
ale_p = ale_p.drop(['Unnamed: 0', 'eventname'], axis=1)
#Adverse Life Experiences (Youth)
ale_y = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_ALE_y_T2_short.csv')
ale_y = ale_y.drop(['Unnamed: 0', 'eventname'], axis=1)
#Peer Experiences Questionnaire (Youth) 
peq = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_PEQ_y_T2_ss.csv')
peq = peq.drop(['Unnamed: 0', 'eventname'], axis=1)
#Discrimination Questionnaire
disc = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_disc_y_T2_short.csv')
disc = disc.drop(['Unnamed: 0', 'eventname'], axis=1)
#Neighborhood Safety Questionnaire
nsc = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_NSC_T2_short.csv')
nsc = nsc.drop(['Unnamed: 0', 'eventname'], axis=1)
#Community Cohesion Scale (Parent)
ccs = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_T2_ccs_p_short.csv')
ccs = ccs.drop(['Unnamed: 0', 'eventname'], axis=1)
#
# %%
#Remove strings from SIT
nan_value = np.nan
cols = list(sit)
for i in cols:
    for row in sit.index:
        if sit[i][row] == "peer condition '-4' was not run":
            sit[i][row] = nan_value
        if sit[i][row] == "peer condition '+2' was not run":
            sit[i][row] = nan_value
# %%
#merge environment dfs into one master df
env = devhx.merge(medhx, on = ['src_subject_id'])
env = env.merge(ale_p, on = ['src_subject_id'])
env = env.merge(ale_y, on = ['src_subject_id'])
env = env.merge(peq, on = ['src_subject_id'])
env = env.merge(disc, on = ['src_subject_id'])
env = env.merge(nsc, on = ['src_subject_id'])
env = env.merge(ccs, on = ['src_subject_id'])
env = env.merge(grouped_demos, on= ['src_subject_id'])
env = env.drop(['age_stop_bedwetting'], axis= 1) #too many NaN values
env_nonan = env.dropna() #shape (2319, 45)
# %%
#merge anat and cog dfs into one master df
abcd = mid_behav.merge(mid_brain, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(sst_behav, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(sst_brain, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(enb_brain, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(enb_behav, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(gdt, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(sit, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(nihtb, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(lmt, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(anat, on= ['src_subject_id', 'eventname'])
abcd = abcd.merge(env, on = ['src_subject_id'])
abcd.shape #(1971, 540)
abcd['group'].sum() #HR group = 947
# %%
####
#Drop NaN values from df
abcd_nonan = abcd.dropna()
abcd_nonan.shape # (1180, 540)
abcd_nonan['group'].sum() #HR group = 566
abcd_nonan.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_uniform_abcd_nonan.csv')

# %%
abcd_nonan['group'].sum()
# %%
env_list = list(env_nonan)
anat_list = list(anat)
dem_list = ['sex', 'interview_age', 'p_edu', 'income', 'race', 'group']
# %%
####
#Separate full df into separate dfs for sub-models
####
#Environmental model
env_abcd = abcd_nonan[env_list] #shape (1180, 45)
#save csv of uniform environmental data
env_abcd.to_csv('/Users/madeleineseitz/Desktop/abcd5_uniform_T2_env_abcd_nonan.csv')
# %%
####
#Anatomical Model
anat_list = anat_list + dem_list
anat_abcd = abcd_nonan[anat_list]
#save csv of uniform sMRI data
anat_abcd.to_csv('/Users/madeleineseitz/Desktop/abcd5_uniform_T2_anat_abcd_nonan.csv')
# %%
# %%
#Cognitive task model
mid_brain_list = list(mid_brain)
mid_behav_list = list(mid_behav)
mid_behav_list.pop(1)
mid_behav_list.pop(0)
sst_brain_list = list(sst_brain)
sst_brain_list.pop(1)
sst_brain_list.pop(0)
sst_behav_list = list(sst_behav)
sst_behav_list.pop(1)
sst_behav_list.pop(0)
enb_brain_list = list(enb_brain)
enb_brain_list.pop(1)
enb_brain_list.pop(0)
enb_behav_list = list(enb_behav)
enb_behav_list.pop(1)
enb_behav_list.pop(0)
gdt_list = list(gdt)
gdt_list.pop(1)
gdt_list.pop(0)
sit_list = list(sit)
sit_list.pop(1)
sit_list.pop(0)
nihtb_list = list(nihtb)
nihtb_list.pop(1)
nihtb_list.pop(0)
lmt_list = list(lmt)
lmt_list.pop(1)
lmt_list.pop(0)
# %%
cog_list = mid_brain_list + mid_behav_list + sst_brain_list + sst_behav_list + enb_brain_list + enb_behav_list + gdt_list + sit_list + nihtb_list + lmt_list + dem_list
# %%
cog_abcd = abcd_nonan[cog_list]
cog_abcd.shape # shape (1180, 359)
# %%
cog_abcd.to_csv('/Users/madeleineseitz/Desktop/cog_abcd5.csv')
# %%
####
#Cogntiive and Anatomical Model
anat_list.pop(1)
anat_list.pop(0)
anat_cog_list = mid_brain_list + mid_behav_list + sst_brain_list + sst_behav_list + enb_brain_list + enb_behav_list + gdt_list + sit_list + nihtb_list + lmt_list + anat_list
anat_cog_abcd = abcd_nonan[anat_cog_list]
anat_cog_abcd.to_csv('/Users/madeleineseitz/Desktop/abcd5_uniform_T2_anat_cog_abcd_nonan.csv')
# %%
abcd_nonan['group'].sum()
# %%
abcd_hr = abcd_nonan.loc[abcd['group'] == 1]
abcd_lr = abcd_nonan.loc[abcd['group'] == 0]
# %%
abcd_hr['interview_age'].mean() #142.212
# %%
abcd_lr['interview_age'].mean() #142.425
# %%
male_count = 0
for row in abcd_lr.index:
    if abcd_lr['sex'][row] == 2:
        male_count += 1
print(male_count) # LR male count = 316; female count = 278, HR male count = 288; female count = 298
# %%
inc_count = 0
for row in abcd_hr.index:
    if abcd_hr['income'][row] == 1.0:
        inc_count += 1
print(inc_count)
# %%
edu_count = 0
for row in abcd_lr.index:
    if abcd_lr['p_edu'][row] == 3:
        edu_count += 1
print(edu_count)
# %%
abcd_hr['ple_y_ss_affected_bad_sum'].mean()
# %%
ids = list(abcd_nonan['src_subject_id'])
# %%
pps = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/PQBC_prepped/pps_refined_LR_T2.csv')
pps_small = pps
for row in pps.index: 
    if pps['src_subject_id'][row] not in ids: 
        pps_small = pps_small.drop([row])
# %%
pps_small.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_pqbc_ml_sample.csv')
# %%
