# Preprocessing Brain and Behavioral Cognitive Task Data

Selected data from the ABCD Cognitive task battery are preprocessed in preparation for ML pipeline and performing statistical analysis to determine presence of significant differences in performance between high and low psychosis risk groups (predetermined in pps_hr_lr_groups script). 

Tasks selected are:
- Little Man Task (follow-up 2 data only)
- Game of Dice Task
- Monetary Incentive Delay Task -- behavioral and fMRI data (follow-up 2 data only)
- Social Influence Task
- NIH Toolbox (follow-up 2 data only)
- Emotional N-Back Task -- behavioral and fMRI data (follow-up 2 data only)
- Stop Signal Task -- behavioral and fMRI data (follow-up 2 data only)

Beta weight parameters were selected based on brain regions relevant to psychosis risk. The most descriptive behavioral parameters were selected for each task. DFs were filtered to include only data from follow-up 2. Each DF column was evaluated for frequency of NaN values, and columns with particularly high NaN values were dropped from the DF (noted in script).

# Preprocessing Structural Scan Data

A subset of ABCD structural scan data are preprocessed in preparation for the ML pipeline (csv: abcd_smrip10201, script: structural_data_preprocessing)
Cortical thickness and cortical volume of right and left Desikan-Killiany regions, in addition to hemisphere and total mean values collected at ABCD follow-up 2 are included. Previous literature has shown that participants at risk for psychosis generally, as well as ABCD psychosis risk participants specifically, exhibit structural brain abnormalities.

# Preprocessing Environmental Data (2/3/25)
1. Peer Experiences Questionnaire
- categorical variables (individual questions) left out in favor of summary scores due to poor variance
2. Discrimination Measure 
- categorical variables (individual questions) left out in favor of summary scores due to poor variance
3. Parental Monitoring Scale
- categorical variables (individual questions) left out in favor of summary scores due to poor variance
4. SRPF
- categorical variables (individual questions) left out in favor of summary scores due to poor variance