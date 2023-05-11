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

plot_coef <- function(data, xlab, xlim, ylab, xintercept, filename) {
  # Create factor levels for Treatment column
  data$Treatment <- factor(data$Treatment, levels = rev(data$Treatment))
  data = data %>% filter(!grepl('Intercept', Treatment))
  
  # Create plot
  p <- ggplot(data, aes(y = Treatment, x = Estimate)) +
    geom_point(size = 6, color = 'green4') +
    geom_pointrange(aes(xmin = Estimate - 1.96 * Std..Error, 
                        xmax = Estimate + 1.96 * Std..Error),
                    color = 'green4', position = position_dodge(0.9),
                    linewidth = 3) +   
    theme(plot.title = element_text(hjust = 0.5)) +
    xlim(xlim) +
    theme_classic() +
    xlab(xlab) +
    geom_vline(xintercept = xintercept, linetype = 'dashed') +
    theme(text = element_text(size = 20)) +
    ylab(ylab)
  
  # Return plot object
  return(p)
}





#Load Data
data <- read_csv('./dat/for_r.csv')


#Reshape data to long format by Belief 
# Add UniqueID as a new column
data$UniqueID <- seq_len(nrow(data))

# Melt the data frame to long format, keeping only relevant columns
data_belief <- data %>%
  select(UniqueID, Country, condName, Belief1, Belief2, Belief3, Belief4) %>%
  pivot_longer(
    cols = starts_with("Belief"),
    names_to = "Item",
    values_to = "Belief"
  ) %>%
  # Drop rows with NA values
  drop_na() %>%
  mutate(condName=  relevel(factor(condName), ref="Control"))

#Preregistered belief model 
lme_climate_beliefs <- lmer(Belief ~ condName + (1|Item) + (1|UniqueID) + (1|Country), data = data_belief)

#Preregistered policy model
plot(lme_climate_beliefs)

# Get coefficients, standard errors, and p-values
coefs <- data.frame(coef(summary(lme_climate_beliefs))) %>% 
            rownames_to_column('Treatment') %>%
            mutate('P-Value'=format.pval(.[[6]], digits=4, eps=.0001)) %>%
            arrange(desc(Estimate)) %>% 
            mutate(Treatment = gsub("condName", "", Treatment))%>% 
            select(-'Pr...t..')


coefs %>% mutate_if(is.numeric, round, digits=3) %>%
    write_delim( './out/preregistered/BeliefCoefs.csv', delim = ",")


# Melt the data frame to long format, keeping only relevant columns
data_policy <- data %>%
  select(UniqueID, Country, condName, Policy1,
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


#Preregistered Policy model 
lme_climate_policy <- lmer(Policy ~ condName + (1|Item) + (1|UniqueID) + (1|Country), data = data_policy)
plot(lme_climate_policy)

# Get coefficients, standard errors, and p-values
coefs_policy <- data.frame(coef(summary(lme_climate_policy))) %>% 
            rownames_to_column('Treatment') %>%
            mutate('P-Value'=format.pval(.[[6]], digits=4, eps=.0001)) %>%
            arrange(desc(Estimate)) %>% 
            mutate(Treatment = gsub("condName", "", Treatment))%>% 
            select(-'Pr...t..')


coefs_policy %>% mutate_if(is.numeric, round, digits=3) %>%
    write_delim( './out/preregistered/PolicyCoefs.csv', delim = ",")


# Melt the data frame to long format, keeping only relevant columns
data_share <- data %>%
  select(UniqueID, Country, condName, SHAREcc) %>%
  pivot_longer(
    cols = starts_with("Share"),
    names_to = "Item",
    values_to = "Share"
  ) %>% 
  drop_na() %>%
  mutate(condName=  relevel(factor(condName), ref="Control"))

#Preregistered Sharing model 
lme_climate_share <- glmer(Share ~ condName  + (1|Country), data = data_share, family=binomial())
plot(lme_climate_share)


# Get coefficients, standard errors, and p-values
coefs_share <- data.frame(coef(summary(lme_climate_share))) %>% 
  rownames_to_column('Treatment') %>%
  mutate('P-Value'=format.pval(.[[5]], digits=4, eps=.0001)) %>%
  arrange(desc(Estimate)) %>% 
  mutate(Treatment = gsub("condName", "", Treatment))%>% 
  select(-'Pr...z..')


coefs_share %>% mutate_if(is.numeric, round, digits=3) %>%
  write_delim( './out/preregistered/ShareCoefs.csv', delim = ",")


# Melt the data frame to long format, keeping only relevant columns
data_WEPT <- data %>%
  select(UniqueID, Country, condName, WEPTcc) %>%
  pivot_longer(
    cols = starts_with("WEPT"),
    names_to = "Item",
    values_to = "WEPT"
  ) %>% 
  drop_na() %>%
  mutate(condName=  relevel(factor(condName), ref="Control"))


#Preregistered WEPT model 
clmm_climate_WEPT <- clmm(factor(WEPT) ~ condName + (1|Country), data = data_WEPT)

# Get coefficients, standard errors, and p-values
coefs_WEPT<- data.frame(coef(summary(clmm_climate_WEPT))) %>% 
            rownames_to_column('Treatment') %>%
            mutate('P-value'=format.pval(.[[5]], digits=4, eps=.001)) %>% 
            filter(grepl( 'condName', Treatment)) %>%
            arrange(desc(Estimate)) %>% 
            mutate(Treatment = gsub("condName", "", Treatment))%>% 
            select(-'Pr...z..')

coefs_WEPT %>% mutate_if(is.numeric, round, digits=3) %>%
    write_delim( './out/preregistered/coefs_WEPT.csv', delim = ",")




#plotting

p1 = plot_coef(coefs,  "Belief", c(-6, 6), "", 0,  "./out/belief.png")
p2 = plot_coef(coefs_policy, "Policy Support", c(-6, 6), "", 0, "./out/policy.png")
p3 = plot_coef(coefs_share,  "Social Media Sharing", c(-0.8, 0.8), "", 0, "./out/share.png")
p4 = plot_coef(coefs_WEPT, "Trees Planted", c(-0.5, 0.5), "", 0,  "./out/wept.png")

plot = p1 + p2 + p3 + p4 + plot_annotation(tag_levels=c('A'))
ggsave('./out/preregistered.png', scale=1.5)


