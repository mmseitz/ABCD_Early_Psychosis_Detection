#This script performs minimal preprocessing on ABCD structural brain data (abcd_smrip10201) and creates a master file of participant cortical
#thickness and cortical volume
# %%
#import libraries
import math
import pandas as pd
import scipy.stats as stats
import numpy as np

# %%
#import csv
smri = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Structural_Scans/abcd_smrip10201.csv')

# %%
#extract relevant columns
#ID and logistics/cortical thickness/cortical volume

cols = ['subjectkey', 'interview_age', 'sex', 'eventname']
all_cols = list(smri)
for i in all_cols:
    if 'thick_cdk' in i:
        cols.append(i)
for i in all_cols:
    if 'vol_cdk'in i:
        cols.append(i)
# %%
#pull relevant columns from main df
reduced_smri = smri[cols]

#select subject data from follow-up 2
reduced_smri = reduced_smri.loc[reduced_smri['eventname'] == '2_year_follow_up_y_arm_1']
# %%
#save csv 
reduced_smri.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Structural_Scans/structural_data.csv')