#####xgboost test
library(xgboost)
library(Matrix)
library(caret)

#https://cran.r-project.org/web/packages/xgboost/vignettes/discoverYourData.html
#https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/

####load the data for analysis
#data extraction code is in the "gap size RF" R script
library(dplyr)
setwd("~/Cassie gaps")
gap.polys<-read.table(file="polys_data")

#remove the shape length column
gap.polys<-gap.polys[,2:19]
summary(gap.polys)

#convert streammin to a yes/no, then set variables as categorical
gap.polys$streamcat<-as.factor(ifelse(gap.polys$stream_min==0,1,0))
gap.polys$geology<-as.factor(gap.polys$geology)
gap.polys$geolarge<-as.factor(gap.polys$geolarge)
gap.polys$geolocal<-as.factor(gap.polys$geolocal)

#convert aspect degrees to radians, then apply northness/eastness transform
deg2rad <- function(deg) {(deg * pi) / (180)}
gap.polys$radians<-deg2rad(gap.polys$aspect)
gap.polys$northness<-cos(gap.polys$radians)
gap.polys$eastness<-sin(gap.polys$radians)
gap.polys$logarea<-log10(gap.polys$Shape_Area)

#convert curvatures to hundredths
gap.polys$curve10plan<-gap.polys$curve10plan/100
gap.polys$curve10prof<-gap.polys$curve10prof/100
gap.polys$curve30plan<-gap.polys$curve30plan/100
gap.polys$curve30prof<-gap.polys$curve30prof/100

#split gaps into large and small
gaps.large<-gap.polys[which(gap.polys$Shape_Area>150),]
write.table(gaps.large,file="large.gaps.data")
gaps.small<-anti_join(gap.polys,gaps.large,by="OBJECTID")
write.table(gaps.small,file="small.gaps.data")


#split large gap data into test, training, etc
gaps.large<-read.table("large.gaps.data")

gaps.large$streamcat<-as.factor(gaps.large$streamcat)
gaps.large$geology<-as.factor(gaps.large$geology)
gaps.large$geolarge<-as.factor(gaps.large$geolarge)
gaps.large$geolocal<-as.factor(gaps.large$geolocal)
#gaps.large$soilgroups<-as.factor(gaps.large$soilgroups)

set.seed(916)
#split test/training/valid for large
vt.gaps.large<-gaps.large[sample(nrow(gaps.large), nrow(gaps.large)*.8),]
test.gaps.large<-anti_join(gaps.large,vt.gaps.large,by="OBJECTID")
train.gaps.large<-vt.gaps.large[sample(nrow(vt.gaps.large), nrow(vt.gaps.large)*.8),]
valid.gaps.large<-anti_join(vt.gaps.large,train.gaps.large,by="OBJECTID")

########XGBOOST FOR LARGE GAPS################

#try xgboost using the method on hackerearth
labels.big<-train.gaps.large$logarea
ts_labels.big<-test.gaps.large$logarea
v_labels.big<-valid.gaps.large$logarea

matrix.train.big<-model.matrix(~x+y+curve10prof+curve10plan+curve30prof+curve30plan+
                             geology+geolarge+geolocal+eastness+elevation+
                             northness+slope+twi+dist_river+streamcat+0,data=train.gaps.large)
matrix.test.big<-model.matrix(~x+y+curve10prof+curve10plan+curve30prof+curve30plan+
                                geology+geolarge+geolocal+eastness+elevation+
                                northness+slope+twi+dist_river+streamcat+0,data=test.gaps.large)
matrix.valid.big<-model.matrix(~x+y+curve10prof+curve10plan+curve30prof+curve30plan+
                                 geology+geolarge+geolocal+eastness+elevation+
                                 northness+slope+twi+dist_river+streamcat+0,data=valid.gaps.large)

dtrain.big<-xgb.DMatrix(data=matrix.train.big,label=as.numeric(labels.big))
dtest.big<-xgb.DMatrix(data=matrix.test.big,label=as.numeric(ts_labels.big))
dvalid.big<-xgb.DMatrix(data=matrix.valid.big,label=as.numeric(v_labels.big))


#now try some grid searching to find the best params
#https://www.kaggle.com/camnugent/gradient-boosting-and-parameter-tuning-in-r
#https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret


#start by looking for the max depth and eta (learning rate)
#setting seed once - THIS ASSUMES YOU CARRY OUT ALL GRID SEARCHING AT THE SAME TIME
set.seed(812)
xgb_grid_1 = expand.grid(nrounds=300,
                         eta = c(0.01,0.05,0.1),
                         max_depth=c(5:15),
                         gamma=0,
                         colsample_bytree=1,
                         min_child_weight=1,
                         subsample=1)
xgb_tc<-trainControl(method="cv",
                     number=5)
xgb_train<-train(x=dtrain.big,y=labels.big,trControl = xgb_tc,tuneGrid=xgb_grid_1,method="xgbTree")
xgb_train
# best outcome:    0.05  15        0.3297842  0.3090000  0.2476140

#keep looking for optimal max_depth with the eta at 0.05
xgb_grid_1.5 = expand.grid(nrounds=300,
                         eta = 0.05,
                         max_depth=c(16:20),
                         gamma=0,
                         colsample_bytree=1,
                         min_child_weight=1,
                         subsample=1)
xgb_tc<-trainControl(method="cv",
                     number=5)
xgb_train<-train(x=dtrain.big,y=labels.big,trControl = xgb_tc,tuneGrid=xgb_grid_1.5,method="xgbTree")
xgb_train

#max depth of 15 was the best option

#this time just do 100 rounds to save time, assuming trends are the same, tune min child weight
xgb_grid_2 = expand.grid(nrounds=100,
                         eta = 0.05,
                         max_depth=15,
                         gamma=0,
                         colsample_bytree=1,
                         min_child_weight=c(5,10,15,20,25,50,100,150,200),
                         subsample=1)
xgb_tc<-trainControl(method="cv",
                     number=5)
xgb_train<-train(x=dtrain.big,y=labels.big,trControl = xgb_tc,tuneGrid=xgb_grid_2,method="xgbTree")
xgb_train
#   50               0.3255426  0.3268005  0.2440167

#finally, tune subsample and column sample
xgb_grid_3 = expand.grid(nrounds=100,
                         eta = 0.05,
                         max_depth=15,
                         gamma=0,
                         colsample_bytree=c(0.4,0.6,0.8,1),
                         min_child_weight=50,
                         subsample=c(0.5,0.75,1))
xgb_tc<-trainControl(method="cv",
                     number=5)
xgb_train<-train(x=dtrain.big,y=labels.big,trControl = xgb_tc,tuneGrid=xgb_grid_3,method="xgbTree")
xgb_train
# 1.0               0.75       0.3250385  0.3282735  0.2443460

#control parameters here
params.big<-list(booster="gbtree",objective="reg:linear",eta=0.05,gamma=0,lambda=0,alpha=1,
             max_depth=15,min_child_weight=50,colsample_bytree=1,subsample=0.75,eval_metric="rmse")

#cross validate to calculate best nrounds
xgbcv.big<-xgb.cv(params=params.big,data=dtrain.big,nrounds = 1000,nfold=5,showsd = TRUE,
              stratified=TRUE,print_every_n = 50,early_stopping_rounds = 20)

#Stopping. Best iteration:
#[205]	train-rmse:0.206925+0.001469	test-rmse:0.322478+0.007949

wlist.big=list(train=dtrain.big,test=dvalid.big)

#train the model
set.seed(829)
xgb.train.big<-xgb.train(params=params.big,data=dtrain.big,nrounds=205,early_stopping_rounds = 10,
                     watchlist=wlist.big)
#[205]	train-rmse:0.207107	test-rmse:0.319806

xgb.pred.big<-predict(xgb.train.big,dtest.big)
save(xgb.train.big,file="xgb.train.big")

load(file="xgb.train.big")

visual.big<-ggplot()+
  geom_point(aes(x=ts_labels.big,y=xgb.pred.big),col="gray28")+
  geom_smooth(aes(x=ts_labels.big,y=xgb.pred.big),method="lm",se=FALSE,color="#F75107FF",size=1.5)+
  geom_abline(intercept=0,slope=1,col="black",linetype="dashed")+
  xlab("Measured gap size (log scale)")+
  ylab("Predicted gap size (log scale)")+
  theme_bw(base_size=18)+
  theme(panel.grid.minor = element_blank())
visual.big
measure.big<-lm(xgb.pred.big~ts_labels.big)
summary(measure.big)

#variable importances
library(Ckmeans.1d.dp)
imps.big<-xgb.importance(feature_names = colnames(matrix.train.big),model=xgb.train.big)
xgb.ggplot.importance(imps.big)+
  theme_bw(base_size=18)+
  scale_fill_fish_d(option="Lampris_guttatus")
write.table(imps.big,file="xgb_big_imps")

###PLOTTING IS IN XGB PLOTS SCRIPT##



########XGBOOST FOR SMALL GAPS################

gaps.small<-read.table("small.gaps.data")

gaps.small$streamcat<-as.factor(gaps.small$streamcat)
gaps.small$geology<-as.factor(gaps.small$geology)
gaps.small$geolarge<-as.factor(gaps.small$geolarge)
gaps.small$geolocal<-as.factor(gaps.small$geolocal)

set.seed(814)
#split test/training/valid for small
vt.gaps.small<-gaps.small[sample(nrow(gaps.small), nrow(gaps.small)*.8),]
test.gaps.small<-anti_join(gaps.small,vt.gaps.small,by="OBJECTID")
train.gaps.small<-vt.gaps.small[sample(nrow(vt.gaps.small), nrow(vt.gaps.small)*.8),]
valid.gaps.small<-anti_join(vt.gaps.small,train.gaps.small,by="OBJECTID")

#try xgboost using the method on hackerearth
labels.small<-train.gaps.small$logarea
ts_labels.small<-test.gaps.small$logarea
v_labels.small<-valid.gaps.small$logarea

matrix.train.small<-model.matrix(~x+y+curve10prof+curve10plan+curve30prof+curve30plan+
                                   geology+geolarge+geolocal+eastness+elevation+
                                   northness+slope+twi+dist_river+streamcat+0,data=train.gaps.small)
matrix.test.small<-model.matrix(~x+y+curve10prof+curve10plan+curve30prof+curve30plan+
                                  geology+geolarge+geolocal+eastness+elevation+
                                  northness+slope+twi+dist_river+streamcat+0,data=test.gaps.small)
matrix.valid.small<-model.matrix(~x+y+curve10prof+curve10plan+curve30prof+curve30plan+
                                   geology+geolarge+geolocal+eastness+elevation+
                                   northness+slope+twi+dist_river+streamcat+0,data=valid.gaps.small)

dtrain.small<-xgb.DMatrix(data=matrix.train.small,label=labels.small)
dtest.small<-xgb.DMatrix(data=matrix.test.small,label=ts_labels.small)
dvalid.small<-xgb.DMatrix(data=matrix.valid.small,label=v_labels.small)


#now try some grid searching to find the best params
#https://www.kaggle.com/camnugent/gradient-boosting-and-parameter-tuning-in-r
#https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret


#start by looking for the right eta (learning rate)
set.seed(815)
xgb_grid_1 = expand.grid(nrounds=300,
                         eta = c(0.01,0.05,0.1),
                         max_depth=c(5:15),
                         gamma=0,
                         colsample_bytree=1,
                         min_child_weight=1,
                         subsample=1)
xgb_tc<-trainControl(method="cv",
                     number=5)
xgb_train<-train(x=dtrain.small,y=labels.small,trControl = xgb_tc,tuneGrid=xgb_grid_1,method="xgbTree")
xgb_train
# best outcome:   
# 0.05   8         0.3664686  0.08969795  0.2996944

#this time just do 100 rounds to save time, assuming trends are the same, tune min child weight
xgb_grid_2 = expand.grid(nrounds=100,
                         eta = 0.05,
                         max_depth=8,
                         gamma=0,
                         colsample_bytree=1,
                         min_child_weight=c(5,10,15,20,25,50,100,150,200),
                         subsample=1)
xgb_tc<-trainControl(method="cv",
                     number=5)
xgb_train<-train(x=dtrain.small,y=labels.small,trControl = xgb_tc,tuneGrid=xgb_grid_2,method="xgbTree")
xgb_train
#  100               0.3667491  0.08868864  0.2997857

#finally, tune subsample and column sample
xgb_grid_3 = expand.grid(nrounds=100,
                         eta = 0.05,
                         max_depth=8,
                         gamma=0,
                         colsample_bytree=c(0.4,0.6,0.8,1),
                         min_child_weight=100,
                         subsample=c(0.5,0.75,1))
xgb_tc<-trainControl(method="cv",
                     number=5)
xgb_train<-train(x=dtrain.small,y=labels.small,trControl = xgb_tc,tuneGrid=xgb_grid_3,method="xgbTree")
xgb_train
#       1.0               0.75       0.3665531  0.08971017  0.2996409

#control parameters here
params<-list(booster="gbtree",objective="reg:linear",eta=0.05,gamma=0,lambda=0,alpha=1,
             max_depth=8,min_child_weight=100,colsample_bytree=1,subsample=0.75,eval_metric="rmse")

#cross validate to calculate best nrounds
xgbcv<-xgb.cv(params=params,data=dtrain.small,nrounds = 1000,nfold=5,showsd = TRUE,
              stratified=TRUE,print_every_n = 50,early_stopping_rounds = 20)
#Stopping. Best iteration:
#[339]	train-rmse:0.348461+0.000212	test-rmse:0.365678+0.000713

wlist=list(train=dtrain.small,test=dvalid.small)

#train the model
set.seed(821)
xgb.train.small<-xgb.train(params=params,data=dtrain.small,nrounds=339,early_stopping_rounds = 10,
                     watchlist=wlist)
#Stopping. Best iteration:
#[265]	train-rmse:0.352903	test-rmse:0.364520

save(xgb.train.small,file="xgb.train.small")
load(file="xgb.train.small")
xgb.pred.small<-predict(xgb.train.small,dtest.small)


visual.small<-ggplot()+
  geom_point(aes(x=ts_labels.small,y=xgb.pred.small))+
  geom_smooth(aes(x=ts_labels.small,y=xgb.pred.small),method="lm",se=FALSE,color="#5F988E")+
  #geom_abline(intercept=0,slope=1,col="red")+
  xlab("Measured gap size (log scale)")+
  ylab("Predicted gap size (log scale)")+
  ylim(0.48,2.0)+
  xlim(0.6,2.0)+
  theme_bw(base_size=18)+
  theme(panel.grid.minor=element_blank())
visual.small
measure.small<-lm(xgb.pred.small~ts_labels.small)
summary(measure.small)

#variable importances
imps.small<-xgb.importance(feature_names = colnames(matrix.train.small),model=xgb.train.small)
xgb.ggplot.importance(imps.small)+
  theme_bw(base_size=18)
write.table(imps.small,file="xgb_small_imps")


