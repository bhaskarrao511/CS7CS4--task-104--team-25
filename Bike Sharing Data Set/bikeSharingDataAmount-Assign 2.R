## Assignment 2 Machine learning
# written by Bhaskar Rao
# Last modified - 15 Dec 2018

#install.packages("dplyr")
library(dplyr)
#install.packages("glmnet")
library(glmnet)
#install.packages("randomForest")
library(randomForest)

#Install Package
#install.packages("e1071")

#Load Library
library(e1071)

# Bike Sharing data set
# setting working directory & importing data
setwd("D:\\Study\\Masters\\Machine learning\\Assignment 1\\Bike-Sharing-Dataset")
bikeDataSet <- read.csv("hour.csv",header = TRUE,stringsAsFactors = FALSE) 
View(bikeDataSet)
summary(bikeDataSet)

#splitting the data into smaller segments for data amount analysis
set.seed(100)
testDataRows <- sample.int(n=nrow(bikeDataSet),size = floor(0.1*nrow(bikeDataSet)),replace = FALSE)
testData <- bikeDataSet[testDataRows,]
trainMaster <- bikeDataSet[-testDataRows,]

# preparing Test Data set
#factorising data set
ads_testData <- testData
ads_testData$holiday <- as.factor(testData$holiday)
ads_testData$weekday <- as.factor(testData$weekday)
ads_testData$workingday <- as.factor(testData$workingday)
ads_testData$casual <- as.factor(testData$casual)
ads_testData$registered <- as.factor(testData$registered)
ads_testData$season <- as.factor(testData$season)
ads_testData$weathersit <- as.factor(testData$weathersit)

ads_testData$sunday <- as.character(ads_testData$weekday )
ads_testData$sunday[ads_testData$sunday == "0"] <- "TRUE"
ads_testData$sunday[ads_testData$sunday != "TRUE"] <- "0"
ads_testData$sunday[ads_testData$sunday == "TRUE"] <- "1"
ads_testData$sunday <- as.factor(ads_testData$sunday)

ads_testData$hourSplit <- ads_testData$hr 
ads_testData$hourSplit[(ads_testData$hourSplit %in% c(0,1,2,3,4,5))] <- 1 # 12 am to 5 am
ads_testData$hourSplit[(ads_testData$hourSplit %in% c(6,7,8,9,10,11))] <- 2 # 6 am to 11 am
ads_testData$hourSplit[(ads_testData$hourSplit %in% c(12,13,14,17,16,15))] <- 3 # 12 pm to 5 pm
ads_testData$hourSplit[(ads_testData$hourSplit %in% c(18,19,20,21,22,23))] <- 4 # 6 pm to 11 pm

## making loop for splitting data sets and running models

summaryListLinear <- list()
summaryListRF <- list()
for(i in 1:10){
  
# taking test data
  testDataSplit <- ads_testData
nsampleBikeSet <- sample.int(n=nrow(trainMaster),size = floor(0.1*i*nrow(trainMaster)),replace = FALSE)
#View(nsampleBikeSet)
bikeSplit <- bikeDataSet[nsampleBikeSet,]
#View(bikeSplit)

#checking the histograms if the data broken follows the same distribution as the population
#? look for more tests for checking if the sample is correct or not
#hist(bikeDataSet$cnt)
#hist(bikeSplit$cnt)

#factorising data set
ads_bikeSplit <- bikeSplit
ads_bikeSplit$holiday <- as.factor(bikeSplit$holiday)
ads_bikeSplit$weekday <- as.factor(bikeSplit$weekday)
ads_bikeSplit$workingday <- as.factor(bikeSplit$workingday)
#ads_bikeSplit$casual <- as.factor(bikeSplit$casual)
#ads_bikeSplit$registered <- as.factor(bikeSplit$registered)
ads_bikeSplit$season <- as.factor(bikeSplit$season)
ads_bikeSplit$weathersit <- as.factor(bikeSplit$weathersit)
#View(ads_bikeSplit)

#dplyr::summarise(ads_bikeSplit$workingday, avg = mean(cnt))
#on sunday, the average is the least, hence will keep 
#ads_bikeSplit %>% group_by(weekday) %>% summarise(count =sum(cnt),average = mean(cnt))

# breaking into sunday
ads_bikeSplit$sunday <- as.character(ads_bikeSplit$weekday )
ads_bikeSplit$sunday[ads_bikeSplit$sunday == "0"] <- "TRUE"
ads_bikeSplit$sunday[ads_bikeSplit$sunday != "TRUE"] <- "0"
ads_bikeSplit$sunday[ads_bikeSplit$sunday == "TRUE"] <- "1"
ads_bikeSplit$sunday <- as.factor(ads_bikeSplit$sunday)

# checking for working day
#ds_bikeSplit %>% group_by(workingday) %>% summarise(count =sum(cnt),average = mean(cnt))

#                       count   mean
# 0 !working day       502749    182.
# 1 working day       1147718    194.

# will keep the variable for working day

# checking for the hour if it has any effect on the cycle count
#hourDistribution<-(ads_bikeSplit %>% group_by(hr) %>% summarise(count =sum(cnt),average = mean(cnt)))
#plot(x = hourDistribution$hr,y=hourDistribution$count)

# from the plot, we can see that the rentals are distributed in 5 hr time difference
ads_bikeSplit$hourSplit <- ads_bikeSplit$hr 
ads_bikeSplit$hourSplit[(ads_bikeSplit$hourSplit %in% c(0,1,2,3,4,5))] <- 1 # 12 am to 5 am
ads_bikeSplit$hourSplit[(ads_bikeSplit$hourSplit %in% c(6,7,8,9,10,11))] <- 2 # 6 am to 11 am
ads_bikeSplit$hourSplit[(ads_bikeSplit$hourSplit %in% c(12,13,14,17,16,15))] <- 3 # 12 pm to 5 pm
ads_bikeSplit$hourSplit[(ads_bikeSplit$hourSplit %in% c(18,19,20,21,22,23))] <- 4 # 6 pm to 11 pm

# modelling

#formula <- cnt ~ season + yr + mnth + holiday + workingday + hum + windspeed + weathersit + hourSplit + sunday + atemp 
formula <- cnt ~ season + yr + holiday + hum + windspeed + weathersit + hourSplit + sunday + atemp 

# linear regression
regLinear <- lm(formula = formula,data = ads_bikeSplit)
#summary(regLinear8k)
summaryListLinear[[i]] <- summary(regLinear)
summaryListLinear[[i]][[2]] <- length(nsampleBikeSet)
testDataSplit$predicted <- (predict(regLinear,testDataSplit))
linearMape <- (mean(abs((testDataSplit$cnt-testDataSplit$predicted)/testDataSplit$cnt)))*100
summaryListLinear[[i]][[3]] <- linearMape
summaryListLinear[[i]][[4]] <- summary(regLinear)$adj.r.squared

# test data for random forest
testDataSplitRF <- ads_testData
# random forest
print(paste("split",i,sep = " "))
randomF <- randomForest(cnt ~ season + yr + holiday + hum + windspeed + weathersit + hourSplit + sunday + atemp ,data = ads_bikeSplit,ntree=150,do.trace=5)
plot(randomF)
common <- intersect(names(ads_bikeSplit), names(testDataSplitRF))
for(p in common){ if (class(ads_bikeSplit[[p]]) == "factor") {
  levels(testDataSplitRF[[p]]) <- levels(ads_bikeSplit[[p]]) } }

testDataSplitRF$predicted <- predict(randomF,testDataSplitRF)
RFMape <- mean(abs((testDataSplitRF$cnt-testDataSplitRF$predicted)/testDataSplitRF$cnt) * 100)
summaryListRF[[i]] <- randomF
summaryListRF[[i]][[2]] <- length(nsampleBikeSet)
summaryListRF[[i]][[3]] <- RFMape
}

summaryListSvm <- list()
for(i in 1:10){
  
  # taking test data

  nsampleBikeSet <- sample.int(n=nrow(trainMaster),size = floor(0.1*i*nrow(trainMaster)),replace = FALSE)
 # nsampleBikeSet <- sample.int(n=nrow(trainMaster),size = floor(0.02*nrow(trainMaster)),replace = FALSE)
  #View(nsampleBikeSet)
  bikeSplit <- bikeDataSet[nsampleBikeSet,]
  #View(bikeSplit)
  
  #checking the histograms if the data broken follows the same distribution as the population
  #? look for more tests for checking if the sample is correct or not
  #hist(bikeDataSet$cnt)
  #hist(bikeSplit$cnt)
  
  #factorising data set
  ads_bikeSplit <- bikeSplit
  ads_bikeSplit$holiday <- as.factor(bikeSplit$holiday)
  ads_bikeSplit$weekday <- as.factor(bikeSplit$weekday)
  ads_bikeSplit$workingday <- as.factor(bikeSplit$workingday)
  #ads_bikeSplit$casual <- as.factor(bikeSplit$casual)
  #ads_bikeSplit$registered <- as.factor(bikeSplit$registered)
  ads_bikeSplit$season <- as.factor(bikeSplit$season)
  ads_bikeSplit$weathersit <- as.factor(bikeSplit$weathersit)
  #View(ads_bikeSplit)
  
  #dplyr::summarise(ads_bikeSplit$workingday, avg = mean(cnt))
  #on sunday, the average is the least, hence will keep 
  #ads_bikeSplit %>% group_by(weekday) %>% summarise(count =sum(cnt),average = mean(cnt))
  
  # breaking into sunday
  ads_bikeSplit$sunday <- as.character(ads_bikeSplit$weekday )
  ads_bikeSplit$sunday[ads_bikeSplit$sunday == "0"] <- "TRUE"
  ads_bikeSplit$sunday[ads_bikeSplit$sunday != "TRUE"] <- "0"
  ads_bikeSplit$sunday[ads_bikeSplit$sunday == "TRUE"] <- "1"
  ads_bikeSplit$sunday <- as.factor(ads_bikeSplit$sunday)
  
  # checking for working day
  #ds_bikeSplit %>% group_by(workingday) %>% summarise(count =sum(cnt),average = mean(cnt))
  
  #                       count   mean
  # 0 !working day       502749    182.
  # 1 working day       1147718    194.
  
  # will keep the variable for working day
  
  # checking for the hour if it has any effect on the cycle count
  #hourDistribution<-(ads_bikeSplit %>% group_by(hr) %>% summarise(count =sum(cnt),average = mean(cnt)))
  #plot(x = hourDistribution$hr,y=hourDistribution$count)
  
  # from the plot, we can see that the rentals are distributed in 5 hr time difference
  ads_bikeSplit$hourSplit <- ads_bikeSplit$hr 
  ads_bikeSplit$hourSplit[(ads_bikeSplit$hourSplit %in% c(0,1,2,3,4,5))] <- 1 # 12 am to 5 am
  ads_bikeSplit$hourSplit[(ads_bikeSplit$hourSplit %in% c(6,7,8,9,10,11))] <- 2 # 6 am to 11 am
  ads_bikeSplit$hourSplit[(ads_bikeSplit$hourSplit %in% c(12,13,14,17,16,15))] <- 3 # 12 pm to 5 pm
  ads_bikeSplit$hourSplit[(ads_bikeSplit$hourSplit %in% c(18,19,20,21,22,23))] <- 4 # 6 pm to 11 pm
  
  # modelling
  
  #formula <- cnt ~ season + yr + mnth + holiday + workingday + hum + windspeed + weathersit + hourSplit + sunday + atemp 
  formula <- cnt ~ season + yr + holiday + hum + windspeed + weathersit + hourSplit + sunday + atemp 
  
  
print(i)
testDataSplitsvm <- ads_testData
common <- intersect(names(ads_bikeSplit), names(testDataSplitsvm))
for(p in common){ if (class(ads_bikeSplit[[p]]) == "factor") {
  levels(testDataSplitsvm[[p]]) <- levels(ads_bikeSplit[[p]]) } }
model_svm <- svm(formula,data = ads_bikeSplit )
testDataSplitsvm$predicted <-predict(model_svm,testDataSplitsvm)

SVMMape <- (mean(abs((testDataSplit$cnt-testDataSplitsvm$predicted)/testDataSplit$cnt)))*100

summaryListSvm[[i]] <- model_svm
summaryListSvm[[i]][[2]] <- length(nsampleBikeSet)
summaryListSvm[[i]][[3]] <- SVMMape

}

####################################################################################################
# creating line chart for the linear model
xVector <- c(summaryListLinear[[1]][[2]])
yVector <- c(summaryListLinear[[1]][[3]])

for(i in 2:10){
  xVector <- append(xVector,summaryListLinear[[i]][[2]])
  yVector <- append(yVector,summaryListLinear[[i]][[3]])
}

plot(x=xVector,y=yVector,type = "l",ylab = "MAPE",xlab = "size of data set")

#####################################################################################################
# creating line chart for the random forest model
xVectorRF <- c(summaryListRF[[1]][[2]])
yVectorRF <- c(summaryListRF[[1]][[3]])

for(i in 2:10){
  xVectorRF <- append(xVectorRF,summaryListRF[[i]][[2]])
  yVectorRF <- append(yVectorRF,summaryListRF[[i]][[3]])
}

plot(x=xVectorRF,y=yVectorRF,type = "l",ylab = "MAPE",xlab = "size of data set")
#####################################################################################################3
# Creating line chart for SVM mape

xVectorSvm <- c(summaryListSvm[[1]][[2]])
yVectorSvm <- c(summaryListSvm[[1]][[3]])

for(i in 2:10){
  xVectorSvm <- append(xVectorSvm,summaryListSvm[[i]][[2]])
  yVectorSvm <- append(yVectorSvm,summaryListSvm[[i]][[3]])
}

plot(x=xVectorSvm,y=yVectorSvm,type = "l",ylab = "MAPE",xlab = "size of data set")

##########################################################################33

percentage <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
dataSetPrint <- data.frame(percentage,xVector,yVector,yVectorRF,yVectorSvm)
View(dataSetPrint)
write.csv(dataSetPrint,"ResultsForBike.csv")

RSS <- c(crossprod(regLinear12k$residuals))
MSE <- RSS / length(regLinear12k$residuals)
RMSE <- sqrt(MSE)
View(RMSE)
summary(ads_bikeSplit)
plot(regLinear12k)
