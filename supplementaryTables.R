library(readxl)
library(ggplot2)
library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)
library(knitr)
library(tibble)
library(readr)
library(ordinal)
library(patchwork)

#Load Data
data <- read_csv('../data/data63.csv')

# Add UniqueID as a new column
data$UniqueID <- seq_len(nrow(data))

# Melt the data frame to long format, keeping only relevant columns
data_belief <- data %>%
  select(UniqueID, Country, condName, condition_time_total,
         Belief1, Belief2, Belief3, Belief4) %>%
  pivot_longer(
    cols = starts_with("Belief"),
    names_to = "Item",
    values_to = "Belief"
  ) %>%
  # Drop rows with NA values
  drop_na() %>%
  mutate(condName=  relevel(factor(condName), ref="Control"))

# Melt the data frame to long format, keeping only relevant columns
data_policy <- data %>%
  select(UniqueID, Country, condName, condition_time_total, Policy1,
         Policy2, Policy3, Policy4, Policy5, 
         Policy6, Policy7, Policy8, Policy9) %>%
  pivot_longer(
    cols = starts_with("Policy"),
    names_to = "Item",
    values_to = "Policy"
  ) %>%
  # Drop rows with NA values
  drop_na(Policy, condName) %>%
  mutate(condName=  relevel(factor(condName), ref="Control"))

# Melt the data frame to long format, keeping only relevant columns
data_share <- data %>%
  select(UniqueID, Country, condName, condition_time_total, SHAREcc) %>%
  pivot_longer(
    cols = starts_with("Share"),
    names_to = "Item",
    values_to = "Share"
  ) %>% 
  drop_na() %>%
  mutate(condName=  relevel(factor(condName), ref="Control"))

# table s6 belief predicting wept
s6Model <- clmm(factor(WEPTcc) ~ BELIEFcc + (1|Country), data = data)
summary(s6Model)

# table s7 policy predicting wept
s7Model <- clmm(factor(WEPTcc) ~ POLICYcc + (1|Country), data = data)
summary(s7Model)

# table s8 share predicting wept
s8Model <- clmm(factor(WEPTcc) ~ SHAREcc + (1|Country), data = data)
summary(s8Model)

# table s9 belief:cond predicting wept
s9Model <- clmm(factor(WEPTcc) ~ BELIEFcc:condName + (1|Country), data = data)
summary(s9Model)

# table s10 policy:cond predicting wept
s10Model <- clmm(factor(WEPTcc) ~ POLICYcc:condName + (1|Country), data = data)
summary(s10Model)

# table s11 share:cond predicting wept
s11Model <- clmm(factor(WEPTcc) ~ SHAREcc:condName + (1|Country), data = data)
summary(s11Model)

# table s12 cond + condTime predicting belief
s12Model <- lmer(Belief ~ condName + condition_time_total + (1|Item) + (1|UniqueID) + (1|Country), data = data_belief)
summary(s12Model)

# table s13 cond + condTime predicting policy
s13Model <- lmer(Policy ~ condName + condition_time_total + (1|Item) + (1|UniqueID) + (1|Country), data = data_policy)
summary(s13Model)

# table s14 cond + condTime predicting share
s14Model <- glmer(Share ~ condName + condition_time_total + (1|Country), data = data_share, family=binomial())
summary(s14Model)

#with condition time

data_WEPT <- data %>%
  select(UniqueID, Country, condName, condition_time_total, WEPTcc) %>%
  pivot_longer(
    cols = starts_with("WEPT"),
    names_to = "Item",
    values_to = "WEPT"
  ) %>% 
  drop_na() %>%
  mutate(condName=  relevel(factor(condName), ref="Control"))

# table s15 cond + condTime predicting wept
s15Model <- clmm(factor(WEPT) ~ condName + condition_time_total + (1|Country), data = data_WEPT)
summary(s15Model)

# table s16 cond + condTime + cond*condTime predicting wept
s16Model <- clmm(factor(WEPT) ~ condName + condition_time_total + condName*condition_time_total + (1|Country), data = data_WEPT)
summary(s16Model)

data80 <- read_csv('../data/WEPT_80/data63.csv')
data80$UniqueID <- seq_len(nrow(data80))

data80 <- data80 %>% 
  select(UniqueID, Country, condName, WEPTcc) %>%
  pivot_longer(
    cols = starts_with("WEPT"),
    names_to = "Item",
    values_to = "WEPT"
  ) %>% 
  drop_na() %>%
  mutate(condName=  relevel(factor(condName), ref="Control"))

# table s17 cond predicting wept for 80% accurate wept
s17Model <- clmm(factor(WEPT) ~ condName + (1|Country), data = data80)
summary(s17Model)






