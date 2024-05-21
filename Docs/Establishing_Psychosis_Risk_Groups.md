# Determining Psychosis High- and Low- Risk Groups Within the ABCD Dataset

This projects evaluates data collected across four time points--from baseline through follow-up 3--in the Adolescent Brain and Cognitive Development longitudinal dataset. It strives to determine whether data representative of participant performance on a battery of cognitive tasks can inform an evaluation of psychosis risk in participants, as cognitive symptoms are often the first to appear in adolescents who may go on to develop psychotic disorders. 

The ABCD dataset utilizes a number of scales to determine psychopathology risk, including the ABCD Prodromal Psychosis Scale (Prodromal Questionnaire-Brief Child Version), or PQ-BC. Detailed information about the scale and its validation are outlined in the JAMA Psychiatry paper "Assessment of the Prodromal Questioinnaire-Brief Child Version for Measurement of Self-Reported Psychoticlike Expereinces in Childhood". Although PLEs commonly occur in nonclinical samples of adolescents ("Validation of the PQ-BC.." Fonseca-Pedrero, 2018) Key points listed below:
- Childhood Psychotic-Like Experiences (PLEs), particularly if they are sustained over time, are associated with a greater chance of developing a psychotic disorder later in life
- PLEs at school age are associated with many of the familial, emotional, and cognitive factors that are associated with psychosis symptoms in older populations (and consistent with the dimensionality of psychosis across the lifespan)
- PLE prevalence decreases with age (occurring in approximately 6-10% of the adult general population, and a much greater percentage of the child population. They decrease fairly rapidly after adolescence)
- Higher PQ-BC scores were also associated with greater parent rated KSADS psychotic symptoms
- Lower crystallized and fluid intelligence was associated with higher PQ-BC scores
- Researchers found a strong association between PLEs and internalizing symptoms

Therefore, in this paper, we use the PQ-BC as our key determinant of psychosis risk in the ABCD population. We place greater significance on PLEs reported at the most recent time point, as PLEs are more likely to be indicative of psychopathology risk in older children and in children who experience sustaiend PLEs across time. Moreover, we focused on participants who both reported PLEs and being bothered by the PLE, as bothersome PLEs are more likely to be indicative of psychopathology.

Further esearch conducted on the PQ-BC in "An Item Respose Theory Analysis of the PQ-BC..." (Karcher, 2020) suggests that each item on the PQ-BC provides a great deal of discriminatory information at high trait levels, but may be less reliable at lower trait levels. The analysis reveals seven items in the PQ-BC that are more strongly associated with other indicators of psychosis risk (i.e., cognitive impairments, developmental milestone delays, parent-rated PLEs, family history of psychosis, and anxiety and depression) and more efficient at discriminating between high and low trait individuals. These questions are PQ-BC items 5, 10, 14, 16, 19, 20, and 21. 

PQ-BC totals and 7-item screener totals are calculated for the ABCD dataset in script "calc_pps_totals.py"

### Family History and mean PQ-BC endorsement and distress correlations

(Karcher et al., 2018)
- Family history of psychotic disorder in first degree relatives was associated with higher mean PQ-BC total and distress, whereas a family history of depression or mania was not
- Higher PQ-BC scores were associated with neuropsychological test performance deficits such as working memory, motor and speech developmental milestone delays (particularly motor and speech devleopment) as well as KSADS internalizing composite
- PLE persistence over time is associated with greater odds of developing a psychotic disorder in adulthood

(Karcher et al., 2021)
- Brain volume associated with environmmental risk factors and PLEs
- PLEs associated with urbanicity, deprivation, and toxins
- Parental PLEs measured in Achenbach Self Report
- study examined (parcellated cortical and subcortical Desikan regions) intracranial volume, total cortical brain volume, total subcortical gray matter volume, total surface area, total cortical thickness, and hippocampal volume due to the wealth of research linking hippocampal volume to both environmental risk factors and PLEs
- Associations between PLEs and environmental risk factors: greater total drug offense exposure and less perception of neighborhood safety were associated with greater PLEs; greater deprivation and fewer years in current residence were associated with greater PLEs; lead exposure was the only environmental toxin associated with greater PLEs
- Each of the structural MRI variables (except hippocampal volume) were signifiicantly associated with PLEs. These structural MRI variables were also associated with all of the environmental risk factors that were significantly associated with PLEs including neighborhood safety, all deprivation indices, and lead exposure risk
- some evidence consistent with brain volume partially mediating the association between deprivation (both rate of poverty and overall deprivation), perceptions of neighborhood safety, and lead exposure risk with PLEs

### PQ-BC endorsement across timepoints
The percentage of participants endorsing PQ-BC questions decreases across timepoints, in keeping that the expectation that the incidence of normal childhood psychotic-like experiences decreases as children reach adolescence. Relatively few participants were lost to follow-up from the baseline to follow-up 2, but substantial dropout (4,161 participants) occurred between follow-ups 2 and 3. Therefore, we will use PQ-BC data from follow-up 2 to determine our high- and low- risk groups. Both cognitive test data and PQ-BC data will be taken from follow-up 2. 799 participants (7.67% of follow-up 2 participants) endorsed being bothered by 3+ PQ-BC questions and, on average, endorsed 2.2 screener questions. 