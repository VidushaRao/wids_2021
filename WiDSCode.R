rm(list = ls())

# libraries
library(rpart) 
library(psych,ggplot2) #pairs.panels 
library(cluster) # Gap statistic of Tibshirani et al 
library(tidyverse) 
library(magrittr) 
library(flashClust) 
library(NbClust) 
library(clValid) 
library(ggfortify) 
library(clustree) 
library(dendextend) 
library(factoextra) 
library(FactoMineR)
library(e1071)
library(caret)
library(pROC)
library(MASS)
library(randomForest)
options(scipen = 100)

# set wd
setwd('C:\\Users\\Vidusha\\Documents\\MSA\\WiDS\\widsdatathon2021')

train = read.csv(file = 'TrainingWiDS2021.csv', stringsAsFactors = FALSE, na.strings = 'NA')
valid = read.csv(file = 'UnlabeledWiDS2021.csv', stringsAsFactors = FALSE, na.strings = 'NA')

# merge datasets
merged = rbind(train[,-length(train)], valid)
merged.binary = as.data.frame(c(1:nrow(merged)))
colnames(merged.binary) = 'row_num'

# split into binary vars, numeric vars, and categorical vars
#binary
for (i in colnames(merged))
{
  print(i)
  if (length(unique(merged[[i]])) == 2)
  {
    merged.binary = cbind(merged.binary, merged[i])
  }
}
merged.binary = merged.binary[,-1]

#numeric
merged.numeric = merged %>%
  select_if(is.numeric)
merged.numeric = merged.numeric[,-which(names(merged.numeric) %in% colnames(merged.binary))]
summary(merged.numeric)

#remove binary variables with too low event proportion
summary(merged.binary)
merged.binary = merged.binary[,-1*c(3,6,7,8,9,10,11,12)]

#remove column with all zeros
for (i in colnames(merged))
{
  if (length(unique(merged[[i]])) == 1)
  {
    print(i)
  }
}
#remove ID variables
merged.numeric = merged.numeric[, -which(names(merged.numeric) %in% c('X','encounter_id','hospital_id','readmission_status'))]


#character
merged.character = merged %>%
  select_if(is.character)


# check missingness of char variables
# if missing, create new variable and impute
for (i in colnames(merged.character))
{
  print(i)
  totmissing = sum(is.na(merged.character[[i]]))
  print(totmissing)
  propmissing = totmissing / nrow(merged.character)
  print(propmissing)
  if (totmissing != 0)
  {
    varname = paste(i,'_missing')
    merged.character[,varname] = 0
    
    for (j in 1:nrow(merged.character))
    {
      if (is.na(merged.character[j,i]) == TRUE)
      {
        merged.character[j,i] = 'Missing'
        merged.character[j,varname] = 1
      }
    }
  }
}

#get num var names
numnames = colnames(merged.numeric)

#get distribution info of num vars
num.shape.df = as.data.frame(c())
for (i in numnames)
{
  varname = i
  varmissing = sum(is.na(merged.numeric[,i]))
  varmean = mean(merged.numeric[,i], na.rm = TRUE)
  varmedian = median(merged.numeric[,i], na.rm = TRUE)
  varskew = skewness(merged.numeric[,i], na.rm = TRUE)
  varkurt = kurtosis(merged.numeric[,i], na.rm = TRUE)
  tempdf = data.frame('VarName' = varname,
                      'VarMissing' = varmissing,
                      'VarMean' = varmean,
                      'VarMedian' = varmedian,
                      'VarSkew' = varskew,
                      'VarKurt' = varkurt)
  num.shape.df = rbind(num.shape.df, tempdf)
}

hist(num.shape.df$VarSkew)
hist(num.shape.df$VarKurt)

#remove highly skewed and high kurt num vars
#for (i in 1:nrow(num.shape.df))
#{
#  if (abs(num.shape.df[i,'VarSkew']) > 1 | abs(num.shape.df[i,'VarKurt']) > 1)
#  {
#    merged.numeric = merged.numeric[, -which(names(merged.numeric) %in% num.shape.df[i,'VarName'])]
#  }
#}

for (i in colnames(merged.numeric))
{
  print(i)
  totmissing = sum(is.na(merged.numeric[[i]]))
  print(totmissing)
  propmissing = totmissing / nrow(merged.numeric)
  print(propmissing)
  
  if (propmissing >= 0.5)
  {
    merged.numeric = merged.numeric[,!(names(merged.numeric) %in% c(i))] 
  }
  else
  {
    medimpute = median(merged.numeric[[i]], na.rm = TRUE)
    print(medimpute)
    for (j in 1:nrow(merged.numeric))
    {
      if (is.na(merged.numeric[j,i]) == TRUE)
      {
        merged.numeric[j,i] = medimpute
      }
    }
  }
}



# correlation pca, standardization/scale on numeric vars
merged.pca = prcomp(merged.numeric, scale = TRUE)
summary(merged.pca)
plot(merged.pca$sdev^2, main ='Scaled Correlation PCA',ylab='Eigenvalue')

#create factors with 23 PCs
fact.loadings = merged.pca$rotation[,1:33] %*% diag(merged.pca$sdev[1:33])
fact.scores = merged.pca$x[,1:33] %*% diag(1/merged.pca$sdev[1:33])

varimax.out = varimax(fact.loadings)
rotated.loadings = fact.loadings %*% varimax.out$rotmat
rotated.scores = fact.scores %*% varimax.out$rotmat

# rejoin data
data.bind1 = cbind(merged.character[,1:6], merged.binary)
merged.cleaned = cbind(data.bind1, rotated.scores)

# convert non numeric vars to factor
merged.cleaned$ethnicity = as.factor(merged.cleaned$ethnicity)
merged.cleaned$gender = as.factor(merged.cleaned$gender)
merged.cleaned$hospital_admit_source = as.factor(merged.cleaned$hospital_admit_source)
merged.cleaned$icu_admit_source = as.factor(merged.cleaned$icu_admit_source)
merged.cleaned$icu_stay_type = as.factor(merged.cleaned$icu_stay_type)
merged.cleaned$icu_type = as.factor(merged.cleaned$icu_type)
merged.cleaned$elective_surgery = as.factor(merged.cleaned$elective_surgery)
merged.cleaned$apache_post_operative = as.factor(merged.cleaned$apache_post_operative)
merged.cleaned$intubated_apache = as.factor(merged.cleaned$intubated_apache)
merged.cleaned$ventilated_apache = as.factor(merged.cleaned$ventilated_apache)

train.final = merged.cleaned[1:nrow(train),]
train.final = cbind(train.final, train[,181])

numlist = c(1:33)
varnamelist = c()
for (i in numlist)
{
  tempval = paste('var',i,sep='')
  varnamelist = c(varnamelist,tempval)
}
varnamelist = c(varnamelist, 'diabetes_mellitus')
colnames(train.final) = c(colnames(train.final[,1:10]), varnamelist)

# write to csv
write.csv(train.final, file = 'TrainSetCleaned.csv', row.names = FALSE)

unlabobs = 1:nrow(train)
unlabeled.final = merged.cleaned[-unlabobs,]
varnamelist = varnamelist[-34]
colnames(unlabeled.final) = c(colnames(train.final[,1:10]), varnamelist)

write.csv(unlabeled.final, file = 'UnlabeledSetCleaned.csv', row.names = FALSE)

# read in data
train.final = read.csv(file = 'TrainSetCleaned.csv', stringsAsFactors = TRUE)

# create random split
set.seed(1111)
samplesize = sample(1:nrow(train.final), 80000, replace = FALSE)
train.1 = train.final[samplesize,]
valid.1 = train.final[-samplesize,]

#########################
## Logistic Regression ##
#########################
# logistic regression with every variable
dm.lr = glm(diabetes_mellitus~., data = train.1, family = binomial)
summary(dm.lr)

#get predicted probabilities probabilities
probs = dm.lr %>%
  predict(valid.1, type = 'response')

# create prediction cutoff
predictions = ifelse(probs > 0.5, 1, 0)

# find error
mean(predictions == valid.1$diabetes_mellitus)

# cunfusion matrix and auc
lr.df = data.frame('Predictions' = predictions,
                   'PredProbs' = probs,
                   'Actual' = valid.1$diabetes_mellitus)

confusionMatrix(as.factor(lr.df$Predictions), as.factor(lr.df$Actual))
auc(lr.df$Actual, lr.df$PredProbs)

###################
## Random Forest ##
###################

#create smaller sample
set.seed(1234)
rfsample = sample(1:nrow(train.1), 20000, replace = FALSE)
train.rf = train.1[rfsample,]

dm.rf = randomForest(as.factor(diabetes_mellitus)~., data = train.rf, type = 'class', importance = TRUE)
dm.rf$confusion

#get predicted probabilities probabilities
rf.probs = dm.rf %>%
  predict(valid.1, type = 'prob')

# create prediction cutoff
rf.predictions = ifelse(rf.probs[,2] > 0.5, 1, 0)

# find error
mean(rf.predictions == valid.1$diabetes_mellitus)

# confusion matrix and auc
rf.df = data.frame('Predictions' = rf.predictions,
                   'PredProbs' = rf.probs[,2],
                   'Actual' = valid.1$diabetes_mellitus)

confusionMatrix(as.factor(rf.df$Predictions), as.factor(rf.df$Actual))
auc(rf.df$Actual, rf.df$PredProbs)

