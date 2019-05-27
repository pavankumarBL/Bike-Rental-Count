#Bike Rent Count Prediction#

rm(list = ls())
setwd("C:/Users/pavankumar.bl/Documents/datascience/Edwisor/Project_3/Training_data")
getwd()
# #loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "e1071",
      "DataCombine", "pROC", "doSNOW", "class", "readxl","ROSE","dplyr", "plyr", "reshape","xlsx","pbapply", "unbalanced", "dummies", "MASS" , "gbm" ,"Information", "rpart", "miscTools")

#load libraries
lapply(x, require, character.only = TRUE)
rm(x)
library("lubridate")

#Read Input Data
df=read.csv("day.csv",header=TRUE,strip.white = TRUE,stringsAsFactors = FALSE)
View(df)

###########################################################################
#                  EXPLORING DATA										  #
###########################################################################
head(df,5)
dim(df)

#structure of data or data types
str(df)

#Summary of data 
summary(df)

#looking at the Structure of data Instant is not significant and 
#proper data conversion is needed. 

#extracting the date time year month from date column and store the each variable in Training dataframe
df <- mutate(df,dteday =ymd(`dteday`),
             day = as.integer(day(dteday))
)


#removing the Date variable since we have extracted the day from it which is needed for analysis.
df = subset(df, select = -c(instant, dteday))

#converting the attributes to valid datatype
df$season=as.factor(df$season)
df$yr=as.factor(df$yr)
df$mnth=as.factor(df$mnth)
df$weekday=as.factor(df$weekday)
df$holiday=as.factor(df$holiday)
df$workingday=as.factor(df$workingday)
df$weathersit=as.factor(df$weathersit)


#unique value of each count
apply(df, 2,function(x) length(table(x)))

# From the above EDA and problem statement categorising data in 2 category "continuous" and "catagorical"
cont_vars = c('temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered','cnt')

cata_vars = c('season','yr','mnth','holiday','weekday', 'workingday', 'weathersit')


#########################################################################
#                     Visualizing the data                              #
#########################################################################


#Plot Number of bikes rented Vs. the day.
ggplot(data = df, aes(x = reorder(day,-cnt), y = cnt))+
  geom_bar(stat = "identity")+
  labs(title = "Number of bikes rented Vs. days", x = "Days", y = "Count")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


#Plot Number of bikes rented Vs. the days of the week.
ggplot(data = df, aes(x = reorder(weekday,-cnt), y = cnt))+
  geom_bar(stat = "identity")+
  labs(title = "Number of bikes rented Vs. days", x = "Days of the week", y = "Count")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


#Plot Bikes rented Vs. variation in temperature and humidity
#It can be observed that people rent bikes mostly when temperature in between 0.5 and0.75 normalized temperature
#and between normalized humidity 0.50 and 0.75
ggplot(df,aes(temp,cnt)) + 
  geom_point(aes(color=hum),alpha=0.5)+
  labs(title = "Bikes rented Vs. variation in temperature and hunidity", x = "Normalized temperature", y = "Count")+
  scale_color_gradientn(colors=c('dark blue','blue','light blue','light green','yellow','orange','red')) +
  theme_bw()

#Plot Bikes rented Vs. temperature and weathersite
#Most bikes are rented duing weather site forcast 1
ggplot(data = df, aes(x = temp, y = cnt))+
  geom_point(aes(color=weathersit))+
  labs(title = "Bikes rented Vs. temperature and weathersite", x = "Normalized temperature", y = "Count")+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme_bw()

#PLot Bikes rented Vs. temperature and workingday
#People rent bikes mostly on working weekdays
ggplot(data = df, aes(x = temp, y = cnt))+
  geom_point(aes(color=workingday))+
  labs(title = "Bikes rented Vs. temperature and workingday", x = "Normalized temperature",y = "Count")+
  #  theme(panel.background = element_rect("white"))+
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))+
  theme_bw()


#################################################################
#         				  Missing data							#
#################################################################

missing_val = sum(is.na(df))
print(missing_val)

################################################################
#               Outlier Analysis			              	   #
################################################################

## BoxPlots - Distribution and Outlier Check
# Boxplot for continuous variables
for (i in 1:length(cont_vars))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cont_vars[i]), x = "cnt",group = 1), data = subset(df))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cont_vars[i],x="cnt")+
           ggtitle(paste("Box plot for",cont_vars[i])))
}

# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn6,ncol=2)


#imputing the Outliers By Capping

fun <- function(x){
  qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = T)
  x[ x < (qnt[1] - H) ] <- caps[1]
  x[ x > (qnt[2] + H)] <- caps[2]
  x
}
df$hum<-fun(df$hum)
df$windspeed <-fun( df$windspeed )
df$casual<-fun(df$casual)

View(df)


################################################################
#               Feature Selection                              #
################################################################

#Here we will use corrgram to find corelation

##Correlation plot
#library('corrgram')

corrgram(df,
         order = F,  #we don't want to reorder
         upper.panel=panel.pie,
         lower.panel=panel.shade,
         text.panel=panel.txt,
         main = 'CORRELATION PLOT')

#We can see that the highly corr related vars in plot are marked in dark blue. 
#Dark blue color means highly positive correlation

##------------------ANOVA testset--------------------------##

## ANOVA testset for Categprical variable
summary(aov(formula = cnt~season,data = df))
summary(aov(formula = cnt~yr,data = df))
summary(aov(formula = cnt~mnth,data = df))
summary(aov(formula = cnt~holiday,data = df))
summary(aov(formula = cnt~weekday,data = df))
summary(aov(formula = cnt~workingday,data = df))
summary(aov(formula = cnt~weathersit,data = df))

##----------------VIF---------------------#
df_vif =subset(df, select = c(temp, atemp, hum, windspeed, casual, registered,cnt))
# Split the data into training and test set

set.seed(123)
training.samples <- df_vif$cnt %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df_vif[training.samples, ]
test.data <- df_vif[-training.samples, ]

# Build the model
model1 <- lm(cnt ~., data = train.data)
# Make predictions
predictions <- model1 %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$cnt),
  R2 = R2(predictions, test.data$cnt)
)
car::vif(model1)

################################################################
#               Feature Selection	                     	   #
################################################################

## Dimension Reduction
#temp and atemp have high correlation , so we have excluded the atemp column.
#holiday, weekday, workingday have p>0.05

df = subset(df, select = -c(atemp,holiday,weekday,workingday))

################################################################
#               Feature Scaling		                     	   #
################################################################

# Updating the continuous and catagorical variables		 

cont_vars = c('temp', 'hum', 'windspeed', 'cnt')

cata_vars = c('season','yr','mnth', 'weathersit')
#Normality check
#Checking Data for Continuous Variables

################  Histogram   ##################
hist(df$cnt, col="blue", xlab="Count", main="Histogram for Count")
hist(df$hum, col="orange", xlab="Count", main="Histogram for Humidity")
hist(df$windspeed, col="sky blue", xlab="Count", main="Histogram for Windspeed")
hist(df$temp, col="red", xlab="Count", main="Histogram for Temperature")
#We have seen that our data is mostly normally distributed. Hence, we will go for Standardization.
#Viewing data before Standardization.
head(df)
cont_vars
#Standardization
for(i in cont_vars)
{
  if(i!= "cnt"){
    print(i)
    df[,i] = (df[,i] - mean(df[,i]))/(sd(df[,i]))
  }
}

#Viewing data after Standardization.
head(df)
#Creating dummy variables for categorical variables
library(mlr)
df1 = dummy.data.frame(df, cata_vars)

#Viewing data after adding dummies
head(df1)
################################################################
#          		        Sampling of Data        			   #
################################################################

# #Divide data into trainset and testset using stratified sampling method

set.seed(101)
split_index = createDataPartition(df1$cnt, p = 0.8, list = FALSE)
trainset = df1[split_index,]
testset  = df1[-split_index,]

#Checking df Set Target Class
table(trainset$cnt)
####FUNCTION to calculate MAPE####
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))*100
}
################################################################
#-------------Model Building and finalizing the Best Model-----#
################################################################

#------------------------------------------Decision tree-------------------------------------------#
#Develop Model on training data
fit_DT = rpart(cnt ~., data = trainset, method = "anova")

#Variable importance
fit_DT$variable.importance

#Lets predict for test data
pred_DT_test = predict(fit_DT, testset)
# error Metrics
print(postResample(pred = pred_DT_test, obs = testset$cnt))

summary(fit_DT)$r.squared

#RMSE         Rsquared         MAE 
#583.8309749   0.9179334      482.6800576
#Compute MSE
dt_mse = mean((testset$cnt - pred_DT_test)^2)
print(dt_mse)

#Compute MAPE
dt_mape = MAPE(testset$cnt, pred_DT_test)
print(dt_mape)

#------------------------------------------Linear Regression-------------------------------------------#

#Develop Model on training data
fit_LR = lm(cnt ~ ., data = trainset)

#Lets predict for test data
pred_LR_test = predict(fit_LR, testset)

# For test data 
print(postResample(pred = pred_LR_test, obs = testset$cnt))
#RMSE         Rsquared        MAE 
#71.5507857   0.9987668       37.5159316

#Compute MSE
lr_mse = mean((testset$cnt - pred_LR_test)^2)
print(lr_mse)

#Compute MAPE
lr_mape = MAPE(testset$cnt, pred_LR_test)
print(lr_mape)

#-----------------------------------------Random Forest----------------------------------------------#

#Develop Model on training data
fit_RF = randomForest(cnt~., data = trainset)

#Lets predict for test data
pred_RF_test = predict(fit_RF, testset)

# For test data 
print(postResample(pred = pred_RF_test, obs = testset$cnt))

#Compute MSE
rf_mse = mean((testset$cnt - pred_RF_test)^2)
print(rf_mse)

#Compute MAPE
rf_mape = MAPE(testset$cnt, pred_RF_test)
print(rf_mape)
#--------------------------------------------XGBoost-------------------------------------------#

#Develop Model on training data
fit_XGB = gbm(cnt~., data = trainset, n.trees = 500, interaction.depth = 2)

#Lets predict for test data
pred_XGB_test = predict(fit_XGB, testset, n.trees = 500)

# For test data 
print(postResample(pred = pred_XGB_test, obs = testset$cnt))
#Compute MSE
xgb_mse = mean((testset$cnt - pred_XGB_test)^2)
print(xgb_mse)

#Compute MAPE
xgb_mape = MAPE(testset$cnt, pred_XGB_test)
print(xgb_mape)
#################-------------------------------Viewing summary of all models------------------------------###############
# Create variables
Model_name <- c("Decision Tree","Linear Regression", "Random Forest", "XGBoost" )
MSE <- c(dt_mse, lr_mse, rf_mse, xgb_mse)
MAPE <- c(dt_mape, lr_mape, rf_mape, xgb_mape)

# Join the variables to create a data frame
results <- data.frame(Model_name,MSE,MAPE)
results
