library("randomForest")

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
project_path      <- getwd() 
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 
articulatoryFeatures <- read.table(file=art_features_path, header=FALSE, sep=",")

trainSize <-1000
inputIndexes <- 1:(length(englishAlphabet)*7)
trainDs <- read.table(file = paste(project_path,"/../encodedDs/datasets/",toString(trainSize),"/nettalk.data",sep=""))


outputIndexes <- (length(englishAlphabet)*7+1):(length(englishAlphabet)*7+18)

most_important_indexes <- matrix(NA,ncol = 26,nrow=length(outputIndexes))

for (idx in outputIndexes){
  cat(idx-length(inputIndexes),"\n")
  response_train <- as.factor(trainDs[,idx])
  temp_train <- data.frame(cbind(trainDs[,inputIndexes],response_train))
  
  rf <- randomForest(response_train~., data=temp_train,ntree=500,importance=TRUE)
  
  most_important_indexes[idx-length(inputIndexes),] <- sort(rf$importance[,3],decreasing=TRUE,index.return=TRUE)$ix[1:26]
  
}


hist(ceiling(most_important_indexes/26),breaks=7)

letters <- most_important_indexes-26*3
letters[letters<0 ]  <- 0
letters[letters>26]  <- 0
letters[letters>0 ]  <- 1
rowSums(letters)/26

letters2 <- most_important_indexes-26*2
letters2[letters2<0 ]  <- 0
letters2[letters2>26*3]  <- 0
letters2[letters2>0 ]  <- 1
rowSums(letters2)/26

