##GROUP 1 PREDICTION PROJECT ECON 386 SP20
#Almudena Badiola, Pablo Gabilondo, Andrea del Saz, Eleena Abdul, Ben Sterbach, Daniel Campbell

#PREPROCESSING AND CLEANING THE DATA
```{r}
library(caret)
library(kernlab)
library(rattle)
library(randomForest)
library(e1071)
library(rpart)
library(ggplot2)
library(rrpack)
library(lattice)
```

```{r}
library(readxl)
Hotel_Bookings_Dataset_for_R <- read_excel("~/Documents/USD/Year 1/Spring/ECON386/Hotel Bookings_Dataset for R.xlsx")
data<-Hotel_Bookings_Dataset_for_R
str(data)
summary(data)
dim(data)
```

```{r}
data$Meal[data$Meal=="Undefined"]<-NA
data$Distribution_Channel[data$Distribution_Channel=="Undefined"]<-NA
data<-na.omit(data)
data$Hotel<-as.factor(data$Hotel)
data$Meal<-as.factor(data$Meal)
data$Distribution_Channel<-as.factor(data$Distribution_Channel)
data$Room<-as.factor(data$Room)
dim(data)
str(data)
```
#Data Partitioning
```{r}
str(data)
set.seed(123)
split <- sample(seq(1, 3), size = nrow(data), replace = TRUE, prob = c(.7, .15, .15))
training <- data[split == 1,]
val <- data[split == 2,]
testing <- data[split == 3,]
```

```{r}
dim(training)
dim(testing)
dim(val)
```

#Plots
```{r, echo=FALSE}
##Average Daily Price by Lead Time
qplot(data$Lead_Time,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Weekend Nights
qplot(data$Weekend_Nights,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Week Nights
qplot(data$Week_Night_Stays,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Number of Children
qplot(data$Children,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Number of Adults
qplot(data$Adults,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Meal Type
qplot(data$Meal,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Distribution Channel
qplot(data$Distribution_Channel,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Repeated Guest
qplot(data$Repeated_Guest,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Room Type
qplot(data$Room,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Booking Changes
qplot(data$Booking_Changes,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Days in Waiting List
qplot(data$Days_Waiting_List,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Parking Spaces
qplot(data$Parking_Spaces,data$Average_Daily_price,geom = c("point","smooth"))
```
```{r, echo=FALSE}
##Average Daily Price by Special Requests
qplot(data$Special_req,data$Average_Daily_price,geom = c("point","smooth"))
```
#Part 3: Regression Modeling
##Almudena´s linear regression models
```{r}
M1<-lm(Average_Daily_price~Hotel+Lead_Time+Weekend_Nights+Week_Night_Stays+Adults+Children+Meal+Distribution_Channel+Repeated_Guest+Room+Booking_Changes+Days_Waiting_List+Parking_Spaces+Special_req,training)
summary(M1)
```
M2<-lm(Average_Daily_price~0+Hotel+Lead_Time+Week_Night_Stays+Weekend_Nights+Adults+Children+Meal+Distribution_Channel+Repeated_Guest+Booking_Changes+Days_Waiting_List+Parking_Spaces+Special_req,training)
summary(M2)
```{r}
M3<-lm(log(Average_Daily_price)~0+Hotel+Lead_Time+Week_Night_Stays+Weekend_Nights+Adults+Children+Meal+Distribution_Channel+Repeated_Guest+Booking_Changes+Days_Waiting_List+Parking_Spaces+Special_req,training)
summary(M3)
```
```{r}
M4<-lm(log(Average_Daily_price)~0+Hotel+I(Lead_Time)+I(Lead_Time^2)+I(Week_Night_Stays)+I(Week_Night_Stays^2)+I(Adults)+I(Adults^2)+I(Children)+I(Children^2)+Meal+Distribution_Channel+I(Repeated_Guest)+I(Repeated_Guest^2)+I(Days_Waiting_List)+I(Days_Waiting_List^2)+I(Parking_Spaces)+I(Parking_Spaces^2)+I(Special_req)+I(Special_req^2),training)
summary(M4)
M5<-lm(log(Average_Daily_price)~0+Hotel+Lead_Time+Week_Night_Stays+Adults+Meal+Distribution_Channel+Days_Waiting_List,training)
summary(M5)
```
##Pablo´s linear regression models
```{r}
P0<-lm(Average_Daily_price~Adults,training)
summary(P0)
Pred_P0<-predict(P0,val)
plot(Pred_P0~Adults,val)
E_IN_P0<-(sum(P0$residuals^2)/(length(P0$residuals)-2))^(1/2)
E_OUT_P0<-(sum(Pred_P0-val$Average_Daily_price)^2/(length(val$Average_Daily_price)-2))^(1/2)
E_IN_P0
E_OUT_P0
```
```{r}
P1<-lm(Average_Daily_price~Adults+Hotel,training)
summary(P1)
Pred_P1<-predict(P1,val)
plot(Pred_P1~Adults,val)
E_IN_P1<-(sum(P1$residuals^2)/(length(P1$residuals)-3))^(1/2)
E_OUT_P1<-(sum(Pred_P1-val$Average_Daily_price)^2/(length(val$Average_Daily_price)-3))^(1/2)
E_IN_P1
E_OUT_P1
```
```{r}
P2<-lm(log(Average_Daily_price)~Adults+Hotel+Meal+Lead_Time,training)
summary(P2)
Pred_P2<-predict(P2,val)
plot(Pred_P2~Adults,val)
E_IN_P2<-(sum(P2$residuals^2)/(length(P2$residuals)-5))^(1/2)
E_OUT_P2<-(sum(Pred_P2-val$Average_Daily_price)^2/(length(val$Average_Daily_price)-4))^(1/2)
E_IN_P2
E_OUT_P2
```
```{r}
P3<-lm(log(Average_Daily_price)~0+Adults+Hotel+Meal+Lead_Time,training)
summary(P3)
Pred_P3<-predict(P3,val)
plot(Pred_P3~Adults,val)
E_IN_P3<-(sum(P3$residuals^2)/(length(P3$residuals)-5))^(1/2)
E_OUT_P3<-(sum(Pred_P3-val$Average_Daily_price)^2/(length(val$Average_Daily_price)-5))^(1/2)
E_IN_P3
E_OUT_P3
plot(P3$residuals)
hist(P3$residuals, prob = TRUE)
curve(dnorm(x, mean = mean(P3$residuals), sd = sd(P3$residuals)), col = "darkblue", lwd = 2, add=TRUE)
summary(P3$residuals)
```
##Andrea´s linear regression models
##Pre-Building Analysis
```{r}
cor(data[,11:14])
cor(data[,2:6])
hist(data$Average_Daily_price,prob = TRUE)
hist(data$Lead_Time,probability = TRUE)
```
##Building the model
```{r}
A0<-lm(Average_Daily_price~Adults+Children, training)
summary(A0)
```
```{r}
A1<-lm(Average_Daily_price~Adults+Children+Lead_Time+Special_req+Weekend_Nights,training)
summary(A1)
```
```{r}
A2<-lm(log(Average_Daily_price)~Adults+Children+Lead_Time+Special_req+Weekend_Nights,training)
summary(A2)
```
```{r}
A3<-lm(log(Average_Daily_price)~0+Adults+Children+Lead_Time+Special_req+Weekend_Nights,training)
summary(A3)
```
##Elena´s linear regression models
```{r}
E1<-lm(Average_Daily_price~Lead_Time+Week_Night_Stays+Booking_Changes+Children, training)
summary(E1)
```
```{r}
E2<-lm(log(Average_Daily_price)~Lead_Time+Week_Night_Stays+Booking_Changes+Children, training)
summary(E2)
```
```{r}
E3<-lm(log(Average_Daily_price)~Lead_Time+Week_Night_Stays+Meal+Children+Adults, training)
summary(E3)
```
```{r}
E4<-lm(log(Average_Daily_price)~0+Lead_Time+Weekend_Nights+Meal+Children+Adults, training)
summary(E4)
```
##Daniel´s linear regression models
```{r}
data2<-data
data2$MealBB[data2$Meal=='BB']<-1
data2$MealBB[data2$Meal!='BB']<-0
data2$MealFB[data2$Meal=='FB']<-1
data2$MealFB[data2$Meal!='FB']<-0
data2$MealHB[data2$Meal=='HB']<-1
data2$MealHB[data2$Meal!='HB']<-0
data2$MealSC[data2$Meal=='SC']<-1
data2$MealSC[data2$Meal!='SC']<-0
data2$Meal<-NULL
data2$HotelResortHotel[data2$Hotel=='Resort Hotel']<-1
data2$HotelResortHotel[data2$Hotel!='Resort Hotel']<-0
data2$HotelCityHotel[data2$Hotel=='City Hotel']<-1
data2$HotelCityHotel[data2$Hotel!='City Hotel']<-0
data2$Hotel<-NULL
data2$Distribution_ChannelCorporate[data2$Distribution_Channel=='Corporate']<-1
data2$Distribution_ChannelCorporate[data2$Distribution_Channel!='Corporate']<-0
data2$Distribution_ChannelDirect[data2$Distribution_Channel=='Direct']<-1
data2$Distribution_ChannelDirect[data2$Distribution_Channel!='Direct']<-0
data2$Distribution_ChannelGDS[data2$Distribution_Channel=='GDS']<-1
data2$Distribution_ChannelGDS[data2$Distribution_Channel!='GDS']<-0
data2$Distribution_ChannelTravelAgent[data2$Distribution_Channel=='Travel Agent']<-1
data2$Distribution_ChannelTravelAgent[data2$Distribution_Channel!='Travel Agent']<-0
data2$Distribution_Channel<-NULL
data2$RoomA[data2$Room=='A']<-1
data2$RoomA[data2$Room!='A']<-0
data2$RoomB[data2$Room=='B']<-1
data2$RoomB[data2$Room!='B']<-0
data2$RoomC[data2$Room=='C']<-1
data2$RoomC[data2$Room!='C']<-0
data2$RoomD[data2$Room=='D']<-1
data2$RoomD[data2$Room!='D']<-0
data2$RoomE[data2$Room=='E']<-1
data2$RoomE[data2$Room!='E']<-0
data2$RoomF[data2$Room=='F']<-1
data2$RoomF[data2$Room!='F']<-0
data2$RoomG[data2$Room=='G']<-1
data2$RoomG[data2$Room!='G']<-0
data2$RoomH[data2$Room=='H']<-1
data2$RoomH[data2$Room!='H']<-0
data2$RoomI[data2$Room=='I']<-1
data2$RoomI[data2$Room!='I']<-0
data2$RoomK[data2$Room=='K']<-1
data2$RoomK[data2$Room!='K']<-0
data2$RoomL[data2$Room=='L']<-1
data2$RoomL[data2$Room!='L']<-0
data2$Room<-NULL
set.seed(123)
split2<-sample(seq(1,3),size = nrow(data2),replace = TRUE, prob = c(.7,.15,.15))
training2<-data2[split2==1,]
val2<-data2[split2==2,]
testing2<-data2[split2==3,]
```
```{r}
D1<-lm(Average_Daily_price~ .,data2)
summary(D1)
```
```{r}
D1train<-lm(Average_Daily_price~ ., training2)
summary(D1train)
```
```{r}
predictions<-predict(D1train, val2)
View(predictions)
RMSE=sqrt(sum((predictions-testing$Average_Daily_price)^2)/(length(testing$Average_Daily_price)-31))
RMSE
```
```{r}
D2<-lm(Average_Daily_price~.-HotelCityHotel-Distribution_ChannelTravelAgent-RoomL-RoomK,data2)
summary(D2)
```
```{r}
D2train<-lm(Average_Daily_price~.-HotelCityHotel-Distribution_ChannelTravelAgent-RoomL-RoomK,training2)
summary(D2train)
predictions2<-predict(D2train, val2)
View(predictions2)
RMSE=sqrt(sum((predictions2-val2$Average_Daily_price)^2)/(length(val2$Average_Daily_price)-28))
RMSE
```
```{r}
D3<-lm(Average_Daily_price~ .-HotelCityHotel-Distribution_ChannelTravelAgent-RoomL-RoomK-RoomA-RoomI-RoomD,data2)
summary(D3)
```
```{r}
D3train<-lm(Average_Daily_price~.-HotelCityHotel-Distribution_ChannelTravelAgent-RoomL-RoomK-RoomA-RoomI-RoomD,training2)
summary(D3train)
predictions3<-predict(D3train, val2)
View(predictions3)
RMSE=sqrt(sum((predictions3-testing$Average_Daily_price)^2)/(length(testing$Average_Daily_price)-27))
RMSE
```
##Ben´s linear regression models
```{r}
cor(data[ ,6:15])
hist(data$Average_Daily_price,prob = TRUE)
hist(data$Children,prob=TRUE)
hist(data$Room,prob=TRUE)
```
```{r}
B1<-lm(Average_Daily_price~Children,data)
summary(B1)
```
```{r}
B2<-lm(Average_Daily_price~Week_Night_Stays,data)
summary(B2)
##R-Squared = .0023
```
```{r}
B3<-lm(Average_Daily_price~Children+Week_Night_Stays,data)
summary(B3)
##R-squared = .1258
```
```{r}
Training <- subset(data, data$Week_Night_Stays!="5")
Testing <- subset(data, data$Week_Night_Stays=="5")
View(Testing)
```
```{r}
B4<-lm(Average_Daily_price~Children+Week_Night_Stays,Training)
summary(B4)
```
```{r}
predictions <- predict(B4,Testing)
View(predictions)
sqrt(sum((predictions-Testing$Average_Daily_price)^2)/(length(Testing$Average_Daily_price)-3))
```
```{r}
B5<-lm(Average_Daily_price~Children+Week_Night_Stays,training)
summary(B5)
```
```{r}
B6<-predict(B5,val)
sqrt(sum((B6-val$Average_Daily_price)^2)/(length(val$Average_Daily_price)-3))
```
```{r}
confint(B3)
confint.default(B3)
point_conf_table<-cbind(B3$coefficients, confint(B3))
point_conf_table
exp(point_conf_table)
```
#Part 4 Validation the regression models
##Almudena´s validation
```{r}
E_IN_M5<-(sum(M5$residuals^2)/(length(M5$residuals)-8))^(1/2)
E_IN_M5
Pred_M5<-predict(M5,val)
E_OUT_M5<-(sum(Pred_M5-val$Average_Daily_price)^2/(length(val$Average_Daily_price)-8))^(1/2)
E_OUT_M5
```
##Pablo´s validation
```{r}
E_IN_P3<-(sum(P3$residuals^2)/(length(P3$residuals)-5))^(1/2)
E_IN_P3
Pred_P3<-predict(P3,val)
E_OUT_P3<-(sum(Pred_P3-val$Average_Daily_price)^2/(length(val$Average_Daily_price)-5))^(1/2)
E_OUT_P3
```
##Andrea´s validation
```{r}
E_IN_A4<-(sum(A3$residuals^2)/(length(A3$residuals)-6))^(1/2)
E_IN_A4
Pred_A4<-predict(A3,val)
E_OUT_A4<-(sum(Pred_A4-val$Average_Daily_price)^2/(length(val$Average_Daily_price)-6))^(1/2)
E_OUT_A4
```
##Eleena´s validation
```{r}
E_IN_E4<-(sum(E4$residuals^2)/(length(E4$residuals)-6))^(1/2)
E_IN_E4
Pred_4<-predict(E4,val)
E_OUT_E4<-(sum(Pred_4-val$Average_Daily_price)^2/(length(val$Average_Daily_price)-6))^(1/2)
E_OUT_E4
```
##Daniel´s validation
```{r}
D3train<-lm(Average_Daily_price~.-HotelCityHotel-Distribution_ChannelTravelAgent-RoomL-RoomK-RoomA-RoomI-RoomD,training2)
summary(D3train)
predictionsD3<-predict(D3train, val2)
RMSE=sqrt(sum((predictionsD3-val2$Average_Daily_price)^2)/(length(val2$Average_Daily_price)-27))
RMSE
```
##Ben´s validation
```{r}
B5<-lm(Average_Daily_price~Children+Week_Night_Stays,training)
summary(B5)
```
```{r}
B6<-predict(B5,val)
View(B6)
sqrt(sum((B6-val$Average_Daily_price)^2)/(length(val$Average_Daily_price)-3))
```

#Part 5 Classification
##Almudena´s classification model
```{r}
training$Repeated_Guest<-factor(training$Repeated_Guest)
testing$Repeated_Guest<-factor(testing$Repeated_Guest)
A_CART <- train(Repeated_Guest~Average_Daily_price+Parking_Spaces+Days_Waiting_List+Week_Night_Stays, data = training, method = "rpart",trControl = trainControl("cv", number = 10),tuneLength = 10)
plot(A_CART)
A_CART$bestTune
confusionMatrix(predict(A_CART,testing),testing$Repeated_Guest)
```
##Pablo´s classification model
```{r}
M_Log_Pablo<-glm(Repeated_Guest~Lead_Time+Weekend_Nights+Booking_Changes+Average_Daily_price,data=training,family="binomial")
summary(M_Log_Pablo)
exp(cbind(M_Log_Pablo$coefficients,confint(M_Log_Pablo)))
confusionMatrix(table(predict(M_Log_Pablo, training, type="response") >= 0.25,training$Repeated_Guest == 1))
confusionMatrix(table(predict(M_Log_Pablo, training, type="response") >= 0.10,training$Repeated_Guest == 1))
````

##Andrea´s classification model
```{r}
A2_CART<-train(Hotel~Average_Daily_price+Lead_Time+Adults+Children+Parking_Spaces, data = training, method = "rpart", trControl=trainControl("cv", number=10), tuneLength=10)
plot(A2_CART)
A2_CART$bestTune
confusionMatrix(predict(A2_CART, testing), testing$Hotel)
````
##Eleena´s classification model
```{r}
M_Log_Elena<-glm(Repeated_Guest~Lead_Time+Weekend_Nights+Adults+Parking_Spaces+Average_Daily_price,data=training,family="binomial")
summary(M_Log_Elena)
exp(cbind(M_Log_Elena$coefficients,confint(M_Log_Elena)))
confusionMatrix(table(predict(M_Log_Elena, testing, type="response") >= 0.25,testing$Repeated_Guest == 1))
````

##Daniel´s classification model
```{r}
D1.1<- glm(Repeated_Guest~.-RoomK-Distribution_ChannelTravelAgent-MealSC, data = data2, family = 'binomial')
summary(D1.1)
set.seed(123)
inTrain<-createDataPartition(y=data2$Repeated_Guest, p=.70, list=FALSE)
Training.1<-data2[inTrain,]#stores rows in training set
Testing.1<-data2[-inTrain,]
M_LOG.1<-glm(Repeated_Guest~.-RoomK-Distribution_ChannelTravelAgent-MealSC, data = Training.1, family='binomial')
summary(M_LOG.1)
confusionMatrix(table(predict(M_LOG.1, Training.1, type="response") >= 0.25, Training.1$Repeated_Guest == 1))
confusionMatrix(table(predict(M_LOG.1, Testing.1, type="response") >= 0.25, Testing.1$Repeated_Guest == 1))
M1.2<- glm(Repeated_Guest~.-RoomK-Distribution_ChannelTravelAgent-MealSC-Booking_Changes-RoomA-RoomB-RoomC-RoomD-RoomE-RoomF-RoomG-RoomH-MealFB-Distribution_ChannelGDS-HotelCityHotel, data = data2, family = 'binomial')
summary(M1.2)
M_LOG.2<- glm(Repeated_Guest~.-RoomK-Distribution_ChannelTravelAgent-MealSC-Booking_Changes-RoomA-RoomB-RoomC-RoomD-RoomE-RoomF-RoomG-RoomH-MealFB-Distribution_ChannelGDS-HotelCityHotel, data=Training.1, family='binomial')
summary(M_LOG.2)
confusionMatrix(table(predict(M_LOG.2, Training.1, type="response") >= 0.25, Training.1$Repeated_Guest == 1))
confusionMatrix(table(predict(M_LOG.2, Testing.1, type="response") >= 0.25, Testing.1$Repeated_Guest == 1))
````
```{r}
point_conf_table<-cbind(M1.2$coefficients, confint(M1.2))#look at confidence interval next to point estimates
point_conf_table
exp(point_conf_table)
````

##Ben´s classification model
```{r}
M_LOG_BEN<-glm(Repeated_Guest~Children+Week_Night_Stays+Average_Daily_price, data = training, family = binomial)
summary(M_LOG_BEN)
exp(cbind(M_LOG_BEN$coefficients, confint(M_LOG_BEN)))
confusionMatrix(table(predict(M_LOG_BEN, training, type="response") >= 0.25,training$Repeated_Guest== 1))
confusionMatrix(table(predict(M_LOG_BEN, testing, type="response") >= 0.25,testing$Repeated_Guest == 1))
````
