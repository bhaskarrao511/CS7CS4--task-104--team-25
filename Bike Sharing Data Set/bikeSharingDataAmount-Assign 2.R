## Assignment 2 Machine learning
# written by Bhaskar Rao
# Last modified - 16 Dec 2018

#install.packages("dplyr")
library(dplyr)
#install.packages("glmnet")
library(glmnet)
#install.packages("randomForest")
library(randomForest)
install.packages("corrplot")
library(corrplot)
  
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

# Correlation matrix

# Numerical variable coorelation with count
 
numericColumnData<-bikeDataSet[,c(11,12,13,14,15,16,17)]

numericColumnDataCor <- cor(numericColumnData)
corrplot(numericColumnDataCor, method = "color")

# we found that windspeed is no corellated to cnt, we will remove from the analysis

bikeDataSet$holiday <- as.factor(bikeDataSet$holiday)
bikeDataSet$weekday <- as.factor(bikeDataSet$weekday)
bikeDataSet$workingday <- as.factor(bikeDataSet$workingday)
bikeDataSet$casual <- as.factor(bikeDataSet$casual)
bikeDataSet$registered <- as.factor(bikeDataSet$registered)
bikeDataSet$season <- as.factor(bikeDataSet$season)
bikeDataSet$weathersit <- as.factor(bikeDataSet$weathersit)


#checking the histograms if the data broken follows the same distribution as the population
#? look for more tests for checking if the sample is correct or not
#hist(bikeDataSet$cnt)
#hist(bikeSplit$cnt)


# creating new features

# checking for working day
#ds_bikeSplit %>% group_by(workingday) %>% summarise(count =sum(cnt),average = mean(cnt))

#                       count   mean
# 0 !working day       502749    182.
# 1 working day       1147718    194.

# will keep the variable for working day

#dplyr::summarise(ads_bikeSplit$workingday, avg = mean(cnt))
#on sunday, the average is the least, hence will keep 
#ads_bikeSplit %>% group_by(weekday) %>% summarise(count =sum(cnt),average = mean(cnt))


bikeDataSet$sunday <- as.character(bikeDataSet$weekday )
bikeDataSet$sunday[bikeDataSet$sunday == "0"] <- "TRUE"
bikeDataSet$sunday[bikeDataSet$sunday != "TRUE"] <- "0"
bikeDataSet$sunday[bikeDataSet$sunday == "TRUE"] <- "1"
bikeDataSet$sunday <- as.factor(bikeDataSet$sunday)

# checking for the hour if it has any effect on the cycle count
#hourDistribution<-(ads_bikeSplit %>% group_by(hr) %>% summarise(count =sum(cnt),average = mean(cnt)))
#plot(x = hourDistribution$hr,y=hourDistribution$count)

# from the plot, we can see that the rentals are distributed in 5 hr time difference

bikeDataSet$hourSplit <- bikeDataSet$hr 
bikeDataSet$hourSplit[(bikeDataSet$hourSplit %in% c(0,1,2,3,4,5))] <- 1 # 12 am to 5 am
bikeDataSet$hourSplit[(bikeDataSet$hourSplit %in% c(6,7,8,9,10,11))] <- 2 # 6 am to 11 am
bikeDataSet$hourSplit[(bikeDataSet$hourSplit %in% c(12,13,14,17,16,15))] <- 3 # 12 pm to 5 pm
bikeDataSet$hourSplit[(bikeDataSet$hourSplit %in% c(18,19,20,21,22,23))] <- 4 # 6 pm to 11 pm



# adjusting hyper parameters

## Random forest - number of trees

formula <- cnt ~ season + yr + holiday + hum +  weathersit + hourSplit + sunday + atemp 
randomFHP <- randomForest(cnt ~ season + yr + holiday + hum +  weathersit + hourSplit + sunday + atemp ,data = bikeDataSet,ntree=300,do.trace=5)

plot(randomFHP,main = "Number of Trees vs error",col = "blue", type="l")
# from the plot, number of trees with minimum error is 155, above that error remains constant

## Lambda in Ridge regression (regularization)

# using glmnet function to generate ridge model, we will use cross validation to get the optimum lambda value

yRreg <- bikeDataSet$cnt
xRreg <- (model.matrix(cnt ~ season + yr + holiday + hum +  weathersit + hourSplit + sunday + atemp, bikeDataSet)[,-1])
lambdas <- 10^seq(3, -2, by = -.1)
gmlRRegHP <- cv.glmnet(x=xRreg , y=yRreg,alpha = 0, lambda = lambdas)

plot(gmlRRegHP)
title("Ridge regression Lambda optimization - Cross Validation", line = + 3)

opt_lambda <- gmlRRegHP$lambda.min
# optimum lambda =  0.3162278
# the top of the graph is number of non zero coefficient estimates, that is 12 and that is not changing



#splitting the data into smaller segments for data amount analysis
set.seed(100)
testDataRows <- sample.int(n=nrow(bikeDataSet),size = floor(0.1*nrow(bikeDataSet)),replace = FALSE)
testData <- bikeDataSet[testDataRows,]
trainMaster <- bikeDataSet[-testDataRows,]


## making loop for splitting data sets and running models
summaryListSvm <- list()
summaryListLinear <- list()
summaryListRF <- list()
for(i in c(0.2,1:10)){
  
# taking test data
  testDataSplit <- ads_testData
nsampleBikeSet <- sample.int(n=nrow(trainMaster),size = floor(0.1*i*nrow(trainMaster)),replace = FALSE)
#View(nsampleBikeSet)
# splitting the data into smaller parts, which keep on increasing with the loop
bikeSplit <- bikeDataSet[nsampleBikeSet,]

#ADS creation
ads_bikeSplit <- bikeSplit

# modelling

#formula <- cnt ~ season + yr + mnth + holiday + workingday + hum + windspeed + weathersit + hourSplit + sunday + atemp 
formula <- cnt ~ season + yr + holiday + hum  + weathersit + hourSplit + sunday + atemp 


#####----------------- linear regression------------------######

regLinear <- lm(formula = formula,data = ads_bikeSplit)

# summary of linear regression
summaryListLinear[[i]] <- summary(regLinear)
summaryListLinear[[i]][[2]] <- length(nsampleBikeSet)
testDataSplit$predicted <- (predict(regLinear,testDataSplit))
linearMape <- (mean(abs((testDataSplit$cnt-testDataSplit$predicted)/testDataSplit$cnt)))*100
summaryListLinear[[i]][[3]] <- linearMape
summaryListLinear[[i]][[4]] <- summary(regLinear)$adj.r.squared


#####----------------- Random forest------------------######

# random forest
print(paste("split",i,sep = " "))
randomF <- randomForest(cnt ~ season + yr + holiday + hum +  weathersit + hourSplit + sunday + atemp ,data = ads_bikeSplit,ntree=155,do.trace=5)
plot(randomF)

# test data for random forest
testDataSplitRF <- ads_testData

# Summary RF
common <- intersect(names(ads_bikeSplit), names(testDataSplitRF))
for(p in common){ if (class(ads_bikeSplit[[p]]) == "factor") {
  levels(testDataSplitRF[[p]]) <- levels(ads_bikeSplit[[p]]) } }

testDataSplitRF$predicted <- predict(randomF,testDataSplitRF)
RFMape <- mean(abs((testDataSplitRF$cnt-testDataSplitRF$predicted)/testDataSplitRF$cnt) * 100)
summaryListRF[[i]] <- randomF
summaryListRF[[i]][[2]] <- length(nsampleBikeSet)
summaryListRF[[i]][[3]] <- RFMape



}


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
