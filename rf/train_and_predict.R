library("randomForest")

englishAlphabet <- strsplit("abcdefghijklmnopqrstuvwxyz","")[[1]]
project_path      <- getwd() 
art_features_path <- paste(project_path,"/../encodedDs/articulatoryFeatures/phonems.csv",sep="") 

for (trainSize in c(200,1000,5000)) { 
  
  outputIndexes <- (length(englishAlphabet)*7+1):(length(englishAlphabet)*7+length(articulatoryFeatures))
  
  
  prediction <- matrix(NA,nrow=dim(testDs)[1],ncol=dim(testDs)[2])
  prediction[,inputIndexes] <- as.matrix(testDs[,inputIndexes])
  
  for (idx in outputIndexes){
    cat(idx-length(inputIndexes),"\n")
    response_train <- as.factor(trainDs[,idx])
    temp_train <- data.frame(cbind(trainDs[,inputIndexes],response_train))
    
    response_test <- as.factor(testDs[,idx])
    temp_test  <- data.frame(cbind(testDs[,inputIndexes],response_test))
    pred <- predict(randomForest(response_train~., data=temp_train,ntree=500),temp_test)
    prediction[,idx] <- as.integer(as.matrix(pred))
  }
  
  write.table(prediction, file = paste(project_path,"/predictions/",toString(trainSize),"/nettalk.predict",sep=""), row.names = FALSE, col.names = FALSE)
  
}