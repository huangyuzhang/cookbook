# ========================================================
# title:    Factors Affecting Work Exhaustion
# author:   Yuzhang Huang
# date:     April 2019
# dataset:  Working From Home (Bloom et al, 2014)
# ========================================================

# clear env (if need)
rm(list=ls())

# import libraries
library(haven)
library(stargazer) # to generate better analytical table
library(Hmisc) # for summary statistics

# Import tables
# performance, labor supply, attrition, promotions, reported work satisfaction, 
# demographic information, and survey information on attitudes toward the program (Exhaustion).

# Attrition <- read_dta("google-drive/QMUL/Modules/BUSM160 Experiment for Business Analytics/Working from home-20190121/Attrition.dta")
# Calls <- read_dta("google-drive/QMUL/Modules/BUSM160 Experiment for Business Analytics/Working from home-20190121/Calls.dta")
Exhaustion <- read_dta("google-drive/QMUL/Modules/BUSM160 Experiment for Business Analytics/Working from home-20190121/Exhaustion.dta")
# Performance <- read_dta("google-drive/QMUL/Modules/BUSM160 Experiment for Business Analytics/Working from home-20190121/Performance.dta")
# Promotion <- read_dta("google-drive/QMUL/Modules/BUSM160 Experiment for Business Analytics/Working from home-20190121/Promotion.dta")
# Satisfaction <- read_dta("google-drive/QMUL/Modules/BUSM160 Experiment for Business Analytics/Working from home-20190121/Satisfaction.dta")

# Data Preparation

# creating data frame
df_exh<-data.frame(Exhaustion)

stargazer(data = df_exh, type = "text", align = TRUE,
          title = "Summary Statistics - Exhaustion")

# time
df_exh<-within(df_exh,{treatment<-ave(experiment_expgroup,year_week,FUN=max)})
df_exh<-within(df_exh,{week<-as.numeric(substr(year_week,5,6))})
df_exh<-within(df_exh,{year<-as.numeric(substr(year_week,1,4))})
df_exh<-within(df_exh,{foy <- chron( paste( 1, 1, year, sep= "/" ) )})
df_exh<-within(df_exh,{fy <- day.of.week( 1, 1, year )})
df_exh<-within(df_exh,{dayno <- ( 0 - fy ) %% 7 + 7 * ( week - 1 )})
df_exh<-within(df_exh,{day<-foy + dayno})
df_exh<-within(df_exh,{foy <- as.Date(as.character(foy),format="%m/%d/%Y")})
df_exh<-within(df_exh,{day <- as.Date(as.character(day),format="%m/%d/%Y")})

# change grosswage to original scale (x1000) and tage log
df_exh<-within(df_exh,{grosswage2<-grosswage*1000})


# describe data
stargazer(data = df_exh, type = "text", align = TRUE,
          title = "Summary Statistics - Exhaustion", out = "r_tables/summary_statistics.tex")

# create variable to identify if is treated or control group
# df_exh<-within(df_sat,{expgroup<-ave(expgroup_treatment,personid,FUN=max)}) # this table already has this variable
df_exh<-within(df_exh,{D_group<-ifelse(expgroup==1, "treated", "control")})
stargazer(df_exh, type="text")

# ========================================================
# Descriptive Analysis


# histograms
par(mfrow=c(1,2))
hist(df_exh$exhaustion, xlab="exhaustion", main=NULL)
hist(df_exh$lnexhaustion, xlab="lnexhaustion", main=NULL)

# see the density difference before and after the experiment
densityplot(~ lnexhaustion | D_group, data=subset(df_exh,treatment==0), layout=c(1,2), main="before WFH",
            panel=function(x,...){
              panel.densityplot(x,...)
              panel.abline(v=quantile(x,.5), col.line="red")
              panel.abline(v=mean(x), col.line="green")
            })

densityplot(~ lnexhaustion | D_group, data=subset(df_exh,treatment==1), layout=c(1,2), main="after WFH",
            panel=function(x,...){
              panel.densityplot(x,...)
              panel.abline(v=quantile(x,.5), col.line="red")
              panel.abline(v=mean(x), col.line="green")
            })
# We can see from the density plot the exhasution gap between treatment group and control group has increased after the experiment,
# which means the working from home has decreased employees exhaustion level.
# This is aligned with the analysis later.

stargazer(subset(df_exh,men==0), type="text")
stargazer(subset(df_exh,men==1), type="text")

# ========================================================
# Build Regression models

# make factor variables
df_exh$personid <- factor(df_exh$personid)
df_exh$year_week <- factor(df_exh$year_week)


# create subsets
df<-df_exh
df0<-subset(df_exh,treatment==0) # before
df1<-subset(df_exh,treatment==1) # after

g0<-subset(df_exh,expgroup==0)
g1<-subset(df_exh,expgroup==1)

# test dataset
# test1<-lm(children ~ married, data=Exhaustion)
# test2<-lm(children ~ married, data=df)
# stargazer(test1,test2,type="text",omit=c("personid"))


# simple linear model to have a basic idea
reg_s<-lm(lnexhaustion ~ age+children+married+bedroom+men+grosswage+commute+high_educ+tenure, data=df)
reg_s1<-lm(lnexhaustion ~ age+children+married+bedroom+men+grosswage+commute+high_educ+tenure+experiment_expgroup, data=df)

stargazer(reg_s,reg_s1, type="text", title="0 Exhaustion: simple", omit=c(), align=TRUE)

# exh<-na.omit(df_exh)

# Q: What are the factors affecting employees exhaustion? 
# H1: working from home decreases employees exhaustion
  # Using the DiD model, control the time variation
em1<-lm(lnexhaustion ~ experiment_expgroup + announcement_expgroup + year_week, data=df_exh) # paper benchmark
# em2<-lm(lnexhaustion ~ experiment_expgroup + expgroup + treatment, data=df_exh) # remove announcement_expgroup
em2<-lm(lnexhaustion ~ experiment_expgroup + year_week + personid, data=df_exh) # add control personid

# H2: employees with higher education degree are less exhausted. (control: individual, group, experiment, time)
em3<-lm(lnexhaustion ~ high_educ + experiment_expgroup + year_week + personid, data=df) # this variable won't change during experiment

# H3: employees with higher grosswage are less exhausted (control: individual, group, experiment, time)
em4<-lm(lnexhaustion ~ grosswage + experiment_expgroup + year_week + personid, data=df) # this variable won't change during experiment

# Analysis: intuitively, higher education degree and grosswage are correlated, thus part of the effect of grosswage on exhaustion is driven by their education level
cor1<-lm(experiment_expgroup ~ high_educ, data=df) # this variable won't change during experiment
stargazer(cor1, type="text", omit=c("day","personid","year_week"), align=TRUE)

# H4: male employees are more exhausted than women.
em5<-lm(lnexhaustion ~ men + experiment_expgroup + year_week + personid, data=df) # this variable won't change during experiment
# em7<-lm(lnexhaustion ~ commute + experiment_expgroup + year_week + personid, data=df) # this variable won't change during experiment

em6<-lm(lnexhaustion ~ high_educ + grosswage + men + experiment_expgroup + year_week + personid, data=df) # this variable won't change during experiment

stargazer(em1, em2, em3, em4, em5, em6, type="text", omit=c("day","personid","year_week"), align=TRUE, title="Regression Models", label ="table:models", omit.stat=c("f", "ser"))

# 
# fem1<-felm(lnexhaustion ~ experiment_expgroup + announcement_expgroup | year_week, data=df_exh) # paper benchmark
# fem2<-felm(lnexhaustion ~ experiment_expgroup | year_week, data=df_exh) # remove announcement_expgroup
# fem3<-felm(lnexhaustion ~ experiment_expgroup | year_week + personid, data=df_exh) # add control personid
# 
# # H2: employees with higher education degree are less exhausted. (control: individual, group, experiment, time)
# fem4<-felm(lnexhaustion ~ high_educ | year_week + personid, data=df) # this variable won't change during experiment
# 
# # H3: employees with higher grosswage are less exhausted (control: individual, group, experiment, time)
# fem5<-felm(lnexhaustion ~ grosswage | year_week + personid, data=df) # this variable won't change during experiment
# 
# # Analysis: intuitively, higher education degree and grosswage are correlated, thus part of the effect of grosswage on exhaustion is driven by their education level
# cor1<-lm(high_educ ~ grosswage, data=df) # this variable won't change during experiment
# stargazer(cor1, type="text", omit=c("day","personid","year_week"), align=TRUE)
# 
# 
# # H4: male employees are more exhausted than women.
# fem6<-felm(lnexhaustion ~ men | year_week + personid, data=df) # this variable won't change during experiment
# # em7<-lm(lnexhaustion ~ commute + experiment_expgroup + year_week + personid, data=df) # this variable won't change during experiment
# 
# # stargazer(fem1, fem2, fem3, fem4, fem5, fem6, type="text", omit=c("day","personid","year_week"), align=TRUE) 
# 
# 
