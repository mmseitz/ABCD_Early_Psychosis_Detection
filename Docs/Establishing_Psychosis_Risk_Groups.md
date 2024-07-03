# Determining Psychosis High- and Low- Risk Groups Within the ABCD Dataset

This projects evaluates data collected across four time points--from baseline through follow-up 3--in the Adolescent Brain and Cognitive Development longitudinal dataset. It strives to determine whether data representative of participant performance on a battery of cognitive tasks can inform an evaluation of psychosis risk in participants, as cognitive symptoms are often the first to appear in adolescents who may go on to develop psychotic disorders. 

The ABCD dataset utilizes a number of scales to determine psychopathology risk, including the ABCD Prodromal Psychosis Scale (Prodromal Questionnaire-Brief Child Version), or PQ-BC. Detailed information about the scale and its validation are outlined in the JAMA Psychiatry paper "Assessment of the Prodromal Questioinnaire-Brief Child Version for Measurement of Self-Reported Psychoticlike Expereinces in Childhood". Although PLEs commonly occur in nonclinical samples of adolescents ("Validation of the PQ-BC.." Fonseca-Pedrero, 2018) Key points listed below:
- Childhood Psychotic-Like Experiences (PLEs), particularly if they are sustained over time, are associated with a greater chance of developing a psychotic disorder later in life
- PLEs at school age are associated with many of the familial, emotional, and cognitive factors that are associated with psychosis symptoms in older populations (and consistent with the dimensionality of psychosis across the lifespan)
- PLE prevalence decreases with age (occurring in approximately 6-10% of the adult general population, and in up to 35% of the child population. They decrease fairly rapidly after adolescence)
- Higher PQ-BC scores were also associated with greater parent rated KSADS psychotic symptoms
- Lower crystallized and fluid intelligence was associated with higher PQ-BC scores
- Lower gray matter volume was associated with higher PQ-BC scores
- Researchers found a strong association between PLEs and internalizing symptoms

Therefore, in this paper, we use the PQ-BC as our key determinant of psychosis risk in the ABCD population. We place greater significance on PLEs reported at later time points, as PLEs are more likely to be indicative of psychopathology risk in older children and in children who experience sustaiend PLEs across time. Analyses are done on data from ABCD follow-up 2, as there was substantial participant drop out at follow-up 3 and because the most recent cognitive task and MRI data is contemporaneous with follow-up 2 PQ-BC scores. (refer to PQBC_summary.xlsx)

Participants are marked as high risk if they endorsed 4 or more PQ-BC items at ABCD follow-up 2. PQ-BC scores above 3.884 were associated with family history of psychotic disorders, greater internalizing symptoms, cognitive task perfomrance deficits, and delayed speech and motor developmental milestones (Karcher et al., 2018). Moreover, Leowy et al., 2011, reported that a PQ-B endorsement threshold of 3+ differentiated between adolescents with prodrome and psychosis symtpoms from those without, with specificity increasing with increased reported distress accompanying item endorsement.

The percentage of participants endorsing PQ-BC questions decreases across timepoints, in keeping that the expectation that the incidence of normal childhood psychotic-like experiences decreases as children reach adolescence. Relatively few participants were lost to follow-up from the baseline to follow-up 2, but substantial dropout (4,161 participants) occurred between follow-ups 2 and 3. Therefore, we will use PQ-BC data from follow-up 2 to determine our high- and low- risk groups. Both cognitive test data and PQ-BC data will be taken from follow-up 2. 

### High-Risk Group Breakdown

Note: code for identifying high-risk participants located in PQBC_breakdown.py

Of the 10,414 participants who completed the PQ-BC at ABCD follow-up 2, 1,646 endorsed 4+ PQ-BC items, falling into our "High Risk" group. High risk participants endorsed a mean of 6.9 total PQ-BC items. 90.76% of this High Risk group also endorsed being bothered by at least 1 PQ-BC item, and on average, each participant endorsed being bothered by 3.77 PQ-BC itmes with an average per-participant bother intensity sum of 10.94. Moreover, this breakdown identifies 18.76% of the ABCD population as High Risk, which is consistent with the proportion of participants found to exhibit subthreshold psychotic symptoms in adolescent populations in other studies (Calkins et al., 2014).
(refer to PQBC_summary.xlsx)

When compared against the rest of the ABCD sample, the High Risk group differed significantly from the rest of the ABCD participant population across demographics measurements (household income, highest level of parental education, ethnicity); mental health summary scores (adverse childhood experiences (ACE) experienced, how much they were affected by reported ACEs, overt agression, relational agression, and overt victim); the Childhood Behavioral Checklist (thought sx, internalizing sx, externalizing sx, and total problems); developmental history (maternal prenatal substance use, age at which pt first walked, parent-reported speech and motor development); family medical history (first-degree relatives hx of psychosis sx, grandparents' hx of psychosis sx); and KSADS  (total parent-reported psychosis sx, total parent-reported psychosis dx).

When the HR group was compared to the rest of the ABCD participant population in across anatomical MRI and cognitive task (behavioral and fMRI data), significant differences were found in whole-brain gray matter volume (anatomical scans--Desikan Parcellation), Monetary Incentive Delay task behavioral measures, Stop Signal Task behavioral measures, the Emotional N-Back task, the Game of Dice task, the Social Influence Task, NIH Toolbox tasks, and the Little Man Behavioral Task. 

These findings are consistent with cognitive and gray matter deficits identified in adolescents with psychosis symptoms in the early psychosis literature, as well as with findings reported in Karcher et al.'s analyses of the ABCD dataset. 
(refer to PQBC_health_demos_stats.xlsx adn PQBC_ABCD_tasks_MRI_stats.xlsx)


### Low-Risk Group Breakdown

