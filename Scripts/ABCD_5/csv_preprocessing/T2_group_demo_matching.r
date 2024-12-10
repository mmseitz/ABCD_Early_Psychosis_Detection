# Install and load necessary packages
install.packages("MatchIt")
library(MatchIt)
library(tidyverse)

# Example data
#set.seed(123)
groupHR <- read_csv('/Users/madeleineseitz/Desktop/ABCD_5/R_matchIt/pps_T2_HR_demos.csv')

groupLR <- read_csv('/Users/madeleineseitz/Desktop/ABCD_5/R_matchIt/pps_T2_LR_demos.csv')

# Combine the datasets and create a treatment indicator
combined_data <- rbind(
  data.frame(group = "groupHR", groupHR),
  data.frame(group = "groupLR", groupLR)
)
combined_data$treatment <- ifelse(combined_data$group == "groupHR", 1, 0)

# Perform matching
matchit_formula <- treatment ~ interview_age + sex + race
match <- matchit(matchit_formula, data = combined_data, method = "nearest", ratio = 1)

# Get matched data
matched_data <- match.data(match)

# Separate matched groupLR data
matched_groupLR <- matched_data[matched_data$group == "groupLR", ]

# Check the balance of covariates
summary(match)

# View matched data
head(matched_groupLR)

# Save full dataset and LR matched data
write.csv(matched_data, "/Users/madeleineseitz/Desktop/T2_demos_matched.csv")
write.csv(matched_groupLR, "/Users/madeleineseitz/Desktop/T2_LR_matched.csv")